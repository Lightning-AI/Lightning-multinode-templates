import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.demos import Transformer, WikiText2
from torch.utils.data import DataLoader, random_split


class LanguageDataModule(L.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.vocab_size = 33278

    def prepare_data(self):
        WikiText2(download=True)

    def setup(self, stage):
        dataset = WikiText2()

        # Split data in to train, val, test
        n = len(dataset)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [n - 4000, 2000, 2000])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class LanguageModel(L.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def configure_model(self):
        self.model = Transformer(vocab_size=self.vocab_size)

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

    datamodule = LanguageDataModule(batch_size=20)

    model = LanguageModel(datamodule.vocab_size)

    # Trainer
    trainer = L.Trainer(gradient_clip_val=0.25, max_epochs=2)
    trainer.fit(model, datamodule=datamodule)
    trainer.save_checkpoint("ptl_default.ckpt")

    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
