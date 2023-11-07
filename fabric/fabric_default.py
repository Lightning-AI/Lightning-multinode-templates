import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.demos import Transformer, WikiText2
from torch.utils.data import DataLoader, random_split


def main():
    L.seed_everything(42)

    fabric = L.Fabric()
    fabric.launch()

    # Data
    with fabric.rank_zero_first(local=True):
        WikiText2(download=True)

    dataset = WikiText2(download=False)

    # Split data in to train, val, test
    n = len(dataset)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [n - 4000, 2000, 2000])
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)

    model = Transformer(vocab_size=dataset.vocab_size)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    model.train()
    num_epochs = 2
    for epoch in range(num_epochs):
        for it, batch in enumerate(train_dataloader):
            input, target = batch
            optimizer.zero_grad()
            output = model(input, target)
            loss = F.nll_loss(output, target.view(-1))
            fabric.print(f"epoch/it: {epoch}/{it}, train_loss: {loss}")
            fabric.backward(loss)
            optimizer.step()

    fabric.save("fabric_default.ckpt", {"model": model})


if __name__ == "__main__":
    main()
