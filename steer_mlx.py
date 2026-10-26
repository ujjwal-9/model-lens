"""MLX port of steer.py: steer individual MLP neurons in Qwen3.5-4B on Apple silicon.

Same recipe: for each (layer L, neuron n, dose d) we add

    d * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||

to neuron n's activation, i.e. the input to layer L's down_proj, at every
position (prompt and generated tokens). The median residual norm comes from a
clean (unsteered) pass over the prompt, skipping token 0.

MLX has no forward hooks, so modules are temporarily swapped for a `Hook`
wrapper. The bf16 HF checkpoint is loaded directly (mlx-lm converts it on
load), so ||W_down[:, n]|| is exact rather than read off quantized weights.

Layers and neurons are 0-indexed.

    python steer_mlx.py                                   # default specs + prompts
    python steer_mlx.py --spec L14:N8949:+1 --spec L13:N7956:-1 --prompt "Hi!"
"""

import argparse
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load, stream_generate

MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_SPECS = ["L14:N8949:+1", "L13:N7956:-1"]
DEFAULT_PROMPTS = [
    "Tell me a little about yourself.",
    "Write a short story about a lighthouse keeper.",
    "What should I cook for dinner tonight?",
]


def parse_spec(s):
    try:
        layer, neuron, dose = s.upper().replace("L", "").replace("N", "").split(":")
        return int(layer), int(neuron), float(dose)
    except ValueError:
        raise ValueError(f"bad spec {s!r}, expected L<layer>:N<neuron>:<dose>, e.g. L14:N8949:+1") from None


def encode(tok, prompt, think=False):
    """Single-turn chat prompt -> token ids."""
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        enable_thinking=think,
    )
    return tok.encode(text)


class Hook(nn.Module):
    """Wraps a module: `pre(x)` can rewrite its input, `post(y)` sees its output."""

    def __init__(self, inner, pre=None, post=None):
        super().__init__()
        self.inner = inner
        self.pre, self.post = pre, post
        if hasattr(inner, "is_linear"):  # the decoder loop reads layer.is_linear
            self.is_linear = inner.is_linear

    def __call__(self, x, *args, **kwargs):
        if self.pre is not None:
            x = self.pre(x)
        y = self.inner(x, *args, **kwargs)
        if self.post is not None:
            self.post(y)
        return y


@contextmanager
def hooks(*installs):
    """Temporarily replace parent[key] with Hook(parent[key], pre, post).

    `parent` is a layer list (key = index) or a module (key = attribute name;
    mlx Modules are dicts, so indexing works for both).
    """
    originals = []
    try:
        for parent, key, pre, post in installs:
            originals.append((parent, key, parent[key]))
            parent[key] = Hook(parent[key], pre, post)
        yield
    finally:
        for parent, key, orig in reversed(originals):
            parent[key] = orig


def record(store, key):
    def f(v):
        store[key] = v
        return v
    return f


def lower_median(x):
    """torch.median convention (lower middle value for even counts), so doses match steer.py."""
    return float(np.sort(x)[(len(x) - 1) // 2])


def w_down_norm(layers, L, n):
    if not 0 <= L < len(layers):
        raise ValueError(f"layer {L} out of range (0-{len(layers) - 1})")
    W = layers[L].mlp.down_proj.weight  # [hidden, intermediate]
    if not 0 <= n < W.shape[1]:
        raise ValueError(f"neuron {n} out of range (0-{W.shape[1] - 1})")
    return mx.linalg.norm(W[:, n].astype(mx.float32)).item()


def calibrate(model, layers, input_ids, specs):
    """Clean pass: median resid norm leaving each layer, plus each neuron's natural activations (tokens 1:)."""
    resid, acts, installs = {}, {}, []
    for L in {L for L, _, _ in specs}:
        installs.append((layers[L].mlp, "down_proj", record(acts, L), None))
        installs.append((layers, L, None, record(resid, L)))
    with hooks(*installs):
        model(mx.array(input_ids)[None])
        mx.eval(resid, acts)  # lazy graph: only computes up to the deepest hooked layer
    medians = {L: lower_median(np.array(mx.linalg.norm(r[0, 1:].astype(mx.float32), axis=-1)))
               for L, r in resid.items()}
    neuron_acts = {(L, n): np.array(acts[L][0, 1:, n].astype(mx.float32)) for L, n, _ in specs}
    return medians, neuron_acts


def plan(model, layers, input_ids, specs):
    """Per-prompt steering offsets (`add`, applied to the neuron's activation) plus stats for display."""
    w_norms = {(L, n): w_down_norm(layers, L, n) for L, n, _ in specs}
    medians, neuron_acts = calibrate(model, layers, input_ids, specs)
    rows = []
    for L, n, d in specs:
        a = neuron_acts[(L, n)]
        rows.append(dict(layer=L, neuron=n, dose=d, median_resid=medians[L], w_norm=w_norms[(L, n)],
                         add=d * medians[L] / w_norms[(L, n)],
                         act_min=float(a.min()), act_med=float(np.median(a)), act_max=float(a.max())))
    return rows


def steering_installs(layers, rows):
    deltas = {}
    for r in rows:
        L = r["layer"]
        deltas.setdefault(L, np.zeros(layers[L].mlp.down_proj.weight.shape[1], np.float32))[r["neuron"]] += r["add"]

    def add(delta):
        return lambda x: x + delta.astype(x.dtype)

    return [(layers[L].mlp, "down_proj", add(mx.array(d)), None) for L, d in deltas.items()]


def generate(model, tok, input_ids, max_new_tokens):
    text, resp = "", None
    for resp in stream_generate(model, tok, input_ids, max_tokens=max_new_tokens):  # greedy by default
        text += resp.text
    return text.strip(), resp.generation_tps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", action="append", help="L<layer>:N<neuron>:<dose>, repeatable (default: %s)" % " ".join(DEFAULT_SPECS))
    ap.add_argument("--prompt", action="append", help="repeatable; defaults to a few generic prompts")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--think", action="store_true", help="leave Qwen3.5 thinking mode on")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    specs = [parse_spec(s) for s in (args.spec or DEFAULT_SPECS)]

    model, tok = load(args.model)
    layers = model.model.layers  # the real list; model.layers returns a copy

    for prompt in args.prompt or DEFAULT_PROMPTS:
        input_ids = encode(tok, prompt, args.think)
        rows = plan(model, layers, input_ids, specs)

        print("=" * 100)
        print(f"PROMPT: {prompt}")
        for r in rows:
            print(f"  L{r['layer']} N{r['neuron']} dose {r['dose']:+g}: median|resid| {r['median_resid']:.2f}"
                  f"  |W_down[:,n]| {r['w_norm']:.4f}  -> add {r['add']:+.2f} to act"
                  f"  (clean act on prompt: min {r['act_min']:+.3f} med {r['act_med']:+.3f} max {r['act_max']:+.3f})")

        out, tps = generate(model, tok, input_ids, args.max_new_tokens)
        print(f"\n--- baseline ({tps:.0f} tok/s) ---")
        print(out)

        with hooks(*steering_installs(layers, rows)):
            out, tps = generate(model, tok, input_ids, args.max_new_tokens)
        print("\n--- steered (" + ", ".join(f"L{L} N{n} {d:+g}" for L, n, d in specs) + f") ({tps:.0f} tok/s) ---")
        print(out)
        print()


if __name__ == "__main__":
    main()
