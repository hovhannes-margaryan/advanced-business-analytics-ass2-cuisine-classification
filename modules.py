import torch
import pytorch_lightning as pl
import torch.nn.functional as F


class GeneralMLightningModule(pl.LightningModule):
    def __init__(self, general_model, learning_rate, model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = general_model(**model_kwargs)
        self.hparams.lr=learning_rate

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        print(f"learning rate: self.hparams.lr")
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_acc', acc)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
