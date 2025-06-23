#!/usr/bin/env python3
"""
train_mnist_ddp.py  ― NCCL/DDP 최소 재현
• 4-layer CNN        • MNIST 자동 다운로드
• torchrun / Kubeflow PyTorchJob 둘 다 호환
"""

import os, torch, torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP

# ── 모델 ────────────────────────────────────────────
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x): return self.net(x)

# ── 분산 set-up / tear-down ─────────────────────────
def setup(): dist.init_process_group(backend="nccl")
def cleanup(): dist.destroy_process_group()

# ── 메인 ────────────────────────────────────────────
def main():
    setup()
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)

    # 데이터
    trans = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST("/tmp/mnist", train=True, download=True, transform=trans)
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds, batch_size=64, sampler=sampler, num_workers=2)

    # 모델 + DDP
    model = SmallCNN().cuda()
    model = DDP(model, device_ids=[local])

    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss().cuda()

    for epoch in range(1):                    # 1 epoch-만
        sampler.set_epoch(epoch)
        for step, (x, y) in enumerate(dl):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            if step % 100 == 0 and rank == 0:
                print(f"[{epoch}/{step}] loss={loss.item():.3f}")

    if rank == 0: print("✔️  MNIST DDP run finished")
    cleanup()

if __name__ == "__main__":
    main()