"""Steer individual MLP neurons in Qwen3.5-4B.

For each (layer L, neuron n, dose d) we add

    d * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||

to neuron n's activation, i.e. the input to layer L's down_proj, at every
position (prompt and generated tokens). Net effect: the residual stream gets
pushed by d * median_norm along the unit direction W_down[:, n] / ||W_down[:, n]||.

The median residual norm is measured on a clean (unsteered) pass over the
prompt, skipping token 0 (attention-sink token with outsized norm).

Layers and neurons are 0-indexed (HF module order / intermediate dim index).

    python steer.py                                   # default specs + prompts
    python steer.py --spec L14:N8949:+1 --spec L13:N7956:-1 --prompt "Hi!"
"""

import argparse

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_SPECS = ["L14:N8949:+1", "L13:N7956:-1"]
DEFAULT_PROMPTS = [
    "Tell me a little about yourself.",
    "Write a short story about a lighthouse keeper.",
    "What should I cook for dinner tonight?",
]


def parse_spec(s):
    layer, neuron, dose = s.upper().replace("L", "").replace("N", "").split(":")
    return int(layer), int(neuron), float(dose)


def decoder_layers(model):
    """The text decoder's layer list (the vision tower's blocks have no down_proj)."""
    for m in model.modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) and hasattr(getattr(m[0], "mlp", None), "down_proj"):
            return m
    raise RuntimeError("no decoder layers with mlp.down_proj found")


@torch.no_grad()
def calibrate(model, layers, input_ids, specs):
    """Clean pass: median resid norm leaving each layer, plus each neuron's natural activations (tokens 1:)."""
    resid, acts, hooks = {}, {}, []
    for L in {L for L, _, _ in specs}:
        hooks.append(layers[L].register_forward_hook(
            lambda m, args, out, L=L: resid.__setitem__(L, out[0] if isinstance(out, tuple) else out)))
        hooks.append(layers[L].mlp.down_proj.register_forward_pre_hook(
            lambda m, args, L=L: acts.__setitem__(L, args[0])))
    try:
        model(input_ids=input_ids)
    finally:
        for h in hooks:
            h.remove()
    medians = {L: r[0, 1:].float().norm(dim=-1).median().item() for L, r in resid.items()}
    neuron_acts = {(L, n): acts[L][0, 1:, n].float() for L, n, _ in specs}
    return medians, neuron_acts


def add_steering_hooks(layers, specs, scales):
    def make_hook(n, s):
        def hook(module, args):
            x = args[0].clone()
            x[..., n] += s
            return (x,) + args[1:]
        return hook

    return [layers[L].mlp.down_proj.register_forward_pre_hook(make_hook(n, s)) for (L, n, _), s in zip(specs, scales)]


@torch.no_grad()
def generate(model, tok, input_ids, max_new_tokens):
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False, temperature=None, top_p=None, top_k=None,
    )
    return tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", action="append", help="L<layer>:N<neuron>:<dose>, repeatable (default: %s)" % " ".join(DEFAULT_SPECS))
    ap.add_argument("--prompt", action="append", help="repeatable; defaults to a few generic prompts")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--think", action="store_true", help="leave Qwen3.5 thinking mode on")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    specs = [parse_spec(s) for s in (args.spec or DEFAULT_SPECS)]
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    layers = decoder_layers(model)

    w_norms = {}
    for L, n, _ in specs:
        W = layers[L].mlp.down_proj.weight  # [hidden, intermediate]
        assert L < len(layers) and n < W.shape[1], f"L{L} N{n} out of range ({len(layers)} layers, {W.shape[1]} neurons)"
        w_norms[(L, n)] = W[:, n].float().norm().item()

    for prompt in args.prompt or DEFAULT_PROMPTS:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
            enable_thinking=args.think,
        )
        input_ids = tok(text, return_tensors="pt").input_ids.to(device)

        medians, neuron_acts = calibrate(model, layers, input_ids, specs)
        scales = [d * medians[L] / w_norms[(L, n)] for L, n, d in specs]

        print("=" * 100)
        print(f"PROMPT: {prompt}")
        for (L, n, d), s in zip(specs, scales):
            a = neuron_acts[(L, n)]
            print(f"  L{L} N{n} dose {d:+g}: median|resid| {medians[L]:.2f}  |W_down[:,n]| {w_norms[(L, n)]:.4f}"
                  f"  -> add {s:+.2f} to act  (clean act on prompt: min {a.min():+.3f} med {a.median():+.3f} max {a.max():+.3f})")

        print("\n--- baseline ---")
        print(generate(model, tok, input_ids, args.max_new_tokens))

        hooks = add_steering_hooks(layers, specs, scales)
        try:
            steered = generate(model, tok, input_ids, args.max_new_tokens)
        finally:
            for h in hooks:
                h.remove()
        print("\n--- steered (" + ", ".join(f"L{L} N{n} {d:+g}" for L, n, d in specs) + ") ---")
        print(steered)
        print()


if __name__ == "__main__":
    main()
