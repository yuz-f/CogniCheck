"""Measure training speed (positions per second) of network sizes on this machine, with random data.

  python scripts/net_speed.py --device mps

Each line: architecture, parameters, positions/s for a full training step (forward, backward,
optimizer) at batch 512, with and without half precision (float16 autocast).
"""
import argparse
import sys
import time

import torch

sys.path.insert(0, ".")
from cognicheck import chessnet  # noqa: E402

CONFIGS = [("resnet", 10, 128, 0), ("transformer", 8, 256, 2), ("transformer", 8, 192, 2), ("transformer", 6, 256, 2),
           ("transformer", 6, 192, 2)]


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def speed(dev, arch, blocks, ch, stem, amp, fused, steps=12, bs=512):
    chessnet.FUSED_ATTENTION = fused
    net = chessnet.build(blocks, ch, 0, chessnet.N_HIST, arch, 8, stem).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    x = torch.rand(bs, chessnet.N_PLANES + 2 * chessnet.N_HIST, 8, 8, device=dev).round()
    e, c = torch.zeros(bs, 2, device=dev), torch.zeros(bs, chessnet.N_CTX, device=dev)
    m = torch.rand(bs, chessnet.N_MOVES, device=dev) < 0.01
    m[:, 0] = True
    t = torch.zeros(bs, dtype=torch.long, device=dev)
    for i in range(steps + 3):
        if i == 3:
            sync(dev)
            t0 = time.time()
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            lo, th = net(x, e, m, None, c, aux=True)
            loss = torch.nn.functional.cross_entropy(lo.float(), t) + th.float().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    sync(dev)
    return sum(p.numel() for p in net.parameters()), steps * bs / (time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    dev = torch.device(a.device)
    print(f"torch {torch.__version__}, device {dev}")
    for arch, blocks, ch, stem in CONFIGS:
        name = f"{arch} {blocks}x{ch}" + (f" stem{stem}" if stem else "")
        for amp in (False, True):
            for fused in ((True, False) if arch == "transformer" else (True,)):
                try:
                    n, ps = speed(dev, arch, blocks, ch, stem, amp, fused)
                    print(f"  {name:26s} {n / 1e6:4.1f}M  {'fp16' if amp else 'fp32'}  {'fused' if fused else 'plain'}: {ps:6,.0f} positions/s", flush=True)
                except Exception as ex:  # an unsupported combination on this device
                    print(f"  {name:26s} {'fp16' if amp else 'fp32'} {'fused' if fused else 'plain'}: failed ({type(ex).__name__}: {str(ex)[:80]})", flush=True)
