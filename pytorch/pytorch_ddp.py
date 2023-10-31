import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from lightning.pytorch.demos import Transformer, WikiText2


def main(global_rank, local_rank, world_size):
    torch.manual_seed(42)

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    # create model and move it to GPU with id local_rank
    device_id = local_rank

    # Data
    if local_rank == 0:
        dataset = WikiText2(".", download=True)

    # Split data in to train, val, test
    n = len(dataset)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [n - 4000, 2000, 2000])
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=20,
                                  shuffle=False,
                                  sampler=DistributedSampler(train_dataset))

    model = Transformer(vocab_size=dataset.vocab_size)
    model.to(device_id)

    ddp_model = DDP(model, device_ids=[device_id])

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

    model.train()
    num_epochs = 2
    for epoch in range(num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        for batch in train_dataloader:
            input, target = batch
            input.to(device_id)
            target.to(device_id)
            optimizer.zero_grad()
            output = ddp_model(input, target)
            loss = F.nll_loss(output, target.view(-1))
            print("train_loss", loss)
            loss.backward()
            optimizer.step()

    dist.destroy_process_group()


if __name__ == "__main__":
    # torch distributed will pick this up
    # master_addr = os.environ["MASTER_ADDR"]
    # master_port = os.environ["MASTER_PORT"]

    global_rank = os.environ["NODE_RANK"]
    local_rank = os.environ["LOCAL_RANK"]
    world_size = os.environ["WORLD_SIZE"]

    mp.spawn(main, args=(global_rank, local_rank, world_size), nprocs=torch.cuda.device_count())
