from time import time

import fire
import torch
from x_transformers import AttentionLayers


def bench(name, model, *inputs, **kwinputs):
    # warmup
    for _ in range(3):
        out = model(*inputs, **kwinputs)
        out.mean().backward()

    start = time()
    for _ in range(100):
        out = model(*inputs, **kwinputs)
        out.mean().backward()
    end = time()
    print(f"{name} took {end-start:.2f}s")


def test_flash(style_name, extra_fwd_inputs={}, **kwargs):
    inp = torch.randn(8, 1024, 1024).cuda()
    standard_kwargs = dict(dim=1024, depth=8, heads=1024//64)
    xf_normal = AttentionLayers(**kwargs, **standard_kwargs).cuda()
    xf_flash = AttentionLayers(**kwargs, attn_use_flash_attention=True, **standard_kwargs).cuda()
    xf_flash.load_state_dict(xf_normal.state_dict())

    with torch.autocast('cuda'):
        out_normal = xf_normal(inp, **extra_fwd_inputs)
        out_flash = xf_flash(inp, **extra_fwd_inputs)
        assert torch.allclose(out_normal, out_flash, rtol=1e-2, atol=1e-2)

        # Bench test them.
        bench(f'normal {style_name}', xf_normal, inp, **extra_fwd_inputs)
        bench(f'flash {style_name}', xf_flash, inp, **extra_fwd_inputs)


def test_all_flash_styles():
    test_flash('encoder')
    test_flash('decoder', causal=True)
    test_flash('cross_attend', causal=True, cross_attend=True,
               extra_fwd_inputs=dict(context=torch.randn(8, 2048, 1024).cuda()))


def test_checkpointing():
    inp = torch.randn(8, 1024, 1024).cuda()
    xf_normal = AttentionLayers(dim=1024, depth=8, heads=1024//64, enable_checkpointing=True).cuda()
    xf_normal(inp).mean().backward()


if __name__ == '__main__':
    fire.Fire()