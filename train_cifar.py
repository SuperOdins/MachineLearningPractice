#!/usr/bin/env python3
# train_cifar_ddp.py
import os, torch, torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(): dist.init_process_group("nccl")
def cleanup(): dist.destroy_process_group()

def main():
    setup()
    rank = dist.get_rank(); local = int(os.getenv("LOCAL_RANK",0))
    torch.cuda.set_device(local)

    # 1) 데이터
    trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    ds = datasets.CIFAR10("/tmp/cifar", train=True, download=True, transform=trans)
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)

    # 2) 모델
    model = models.resnet18(num_classes=10).cuda()
    model = DDP(model, device_ids=[local])

    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(dl))
    loss_fn = nn.CrossEntropyLoss().cuda()

    for step, (x,y) in enumerate(dl):
        x,y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        opt.zero_grad(); pred = model(x); loss = loss_fn(pred,y); loss.backward(); opt.step(); sched.step()
        if step%100==0 and rank==0:
            print(f"[{step}/{len(dl)}] loss={loss.item():.3f}")
    if rank==0: print("✔️ CIFAR10 DDP finished")
    cleanup()

if __name__ == "__main__": main()