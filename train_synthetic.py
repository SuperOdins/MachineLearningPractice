#!/usr/bin/env python3
"""
train_synth224_ddp.py
────────────────────────────────────────────────────────────
• 224×224 RGB 랜덤 이미지 10k개 → Tiny ViT-B/16 forward/backward
• 멀티-노드 NCCL, DDP, bf16 혼합 단정도
"""

import os, torch, torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models.vision_transformer import vit_b_16

# ---------- 1. Synthetic Dataset ----------
class RandomImageSet(Dataset):
    def __init__(self, n=10_000, c=1_000):
        self.x = torch.randn(n, 3, 224, 224)
        self.y = torch.randint(0, c, (n,))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

# ---------- 2. DDP Setup / Cleanup ----------
def setup():    dist.init_process_group("nccl")
def cleanup():  dist.destroy_process_group()

# ---------- 3. Main ----------
def main():
    setup()
    rank   = dist.get_rank()
    local  = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)

    # 3-1. 모델 (Tiny ViT-B/16)
    model = vit_b_16(num_classes=1_000).cuda().bfloat16()
    model = DDP(model, device_ids=[local], find_unused_parameters=False)

    # 3-2. 데이터 / 로더
    ds = RandomImageSet()
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds, batch_size=64, sampler=sampler,
                    num_workers=2, pin_memory=True)

    opt  = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss = nn.CrossEntropyLoss().cuda()

    # 3-3. 학습 루프 (1 epoch)
    for step, (xb, yb) in enumerate(dl):
        xb = xb.cuda(non_blocking=True).bfloat16()
        yb = yb.cuda(non_blocking=True)

        opt.zero_grad()
        pred = model(xb)
        l    = loss(pred, yb)
        l.backward()
        opt.step()

        if step % 50 == 0 and rank == 0:
            print(f"[{step}/{len(dl)}] loss={l.item():.3f}")

    if rank == 0:
        print("✔️ Synthetic-224 DDP run finished.")
    cleanup()

if __name__ == "__main__":
    main()