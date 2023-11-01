import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.demos import Transformer, WikiText2
from torch.utils.data import DataLoader, random_split


class LanguageModel(L.LightningModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        WikiText2(download=True)

    def setup(self, stage):
        dataset = WikiText2()
        self.model = Transformer(vocab_size=dataset.vocab_size)

        # Split data in to train, val, test
        n = len(dataset)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [n - 4000, 2000, 2000])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def main():
    L.seed_everything(42)

    model = LanguageModel(batch_size=20)

    # Trainer
    trainer = L.Trainer(gradient_clip_val=0.25, max_epochs=2, strategy="deepspeed_stage_2")
    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    main()
