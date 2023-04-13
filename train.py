import os
import torch
import pytorch_lightning as pl
from torchvision import transforms
from argparse import ArgumentParser
from models import VisionTransformer
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset
from modules import GeneralMLightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

parser = ArgumentParser()
parser.add_argument("--num_classes", default=20, help="number of classes", type=int)
parser.add_argument("--pretrained_model_path",  help="path to the pretrained model", type=str)
parser.add_argument("--train_parquet_path",  help="path to the train parquet", type=str)
parser.add_argument("--validation_parquet_path",  help="path to the validation parquet", type=str)
parser.add_argument("--batch_size", default=128,  help="training and validation batch size", type=int)
parser.add_argument("--device", default="cuda",  help="training and validation batch size", type=str)
parser.add_argument("--image_size", default=224, help="size of the image for training", type=int)
parser.add_argument("--max_epochs", default=500, help="maximum number of epochs to train", type=int)
parser.add_argument("--model_checkpoint_path", help="path to save the trained model checkpoint", type=str)
parser.add_argument("--learning_rate", help="learning rate", type=float)

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device(args.device)
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop((args.image_size, args.image_size),
                                                                       scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.RandomRotation(degrees=90),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                                                               [0.24703223, 0.24348513, 0.26158784])])

    validation_transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                                                                     [0.24703223, 0.24348513, 0.26158784])])

    train_dataset = CustomDataset(args.train_parquet_path, train_transform)
    validation_dataset = CustomDataset(args.validation_parquet_path, train_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  drop_last=True,  num_workers=4)
    validation_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,  shuffle=False,
                                       drop_last=False, num_workers=4)

    trainer = pl.Trainer(default_root_dir=args.model_checkpoint_path,
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=args.max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True,
                                                    mode="max", monitor="val_acc")])

    trainer.logger._log_graph = False
    trainer.logger._default_hp_metric = None

    if os.path.isfile(args.pretrained_model_path):
        model = GeneralMLightningModule.load_from_checkpoint(args.pretrained_model_path)
    else:
        model = GeneralMLightningModule(VisionTransformer, args.learning_rate,
                                        {"num_classes": args.num_classes})

    trainer.fit(model, train_dataloader, validation_dataloader)

