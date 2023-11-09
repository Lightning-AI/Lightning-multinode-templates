import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from lightning.pytorch.demos import Transformer, WikiText2


def main(local_rank, global_rank, world_size, device):
    torch.manual_seed(42)

    backend = "nccl" if device == "cuda" else "gloo"
    if global_rank == -1:
        global_rank = local_rank

    dist.init_process_group(backend=backend, rank=global_rank, world_size=world_size)

    device_id = local_rank

    if local_rank == 0:
        WikiText2(download=True)

    dist.barrier()

    dataset = WikiText2(download=False)

    n = len(dataset)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [n - 4000, 2000, 2000])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=20,
        shuffle=False,
        sampler=DistributedSampler(train_dataset)
    )

    model = Transformer(vocab_size=dataset.vocab_size)
    if device == "cuda":
        model.to(device_id)

    device_ids = [device_id] if device == "cuda" else None
    ddp_model = DDP(model, device_ids=device_ids)

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

    model.train()
    num_epochs = 2
    for epoch in range(num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        for it, batch in enumerate(train_dataloader):
            input, target = batch
            if device == "cuda":
                input = input.to(device_id)
                target = target.to(device_id)
            optimizer.zero_grad()
            output = ddp_model(input, target)
            loss = F.nll_loss(output, target.view(-1))
            if global_rank == 0:
                print(f"epoch/it: {epoch}/{it}, train_loss {float(loss)}")
            loss.backward()
            optimizer.step()

    if global_rank == 0:
        state = {
            'model': ddp_model.module.state_dict(),
        }
        torch.save(state, "pytorch_ddp.ckpt")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "6006"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nprocs = torch.cuda.device_count() if device == "cuda" else 1

    global_rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", nprocs))

    if global_rank == -1:
        mp.spawn(main, args=(global_rank, world_size, device), nprocs=nprocs)
    else:
        main(local_rank, global_rank, world_size, device)
