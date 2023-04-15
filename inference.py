import torch
from torchvision import transforms
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset
from modules import GeneralMLightningModule
from statistics import mean
import tqdm

parser = ArgumentParser()
parser.add_argument("--pretrained_model_path",  help="path to the pretrained model", type=str)
parser.add_argument("--test_parquet_path",  help="path to the train parquet", type=str)
parser.add_argument("--device", default="cuda",  help="training and validation batch size", type=str)
parser.add_argument("--batch_size", default=128,  help="training and validation batch size", type=int)
parser.add_argument("--image_size", default=224, help="size of the image for training", type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device(args.device)
    test_transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                                                                    [0.24703223, 0.24348513, 0.26158784])])
    test_dataset = CustomDataset(args.test_parquet_path, test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                       drop_last=False, num_workers=4)

    model = GeneralMLightningModule.load_from_checkpoint(args.pretrained_model_path)
    model.eval()

    with torch.no_grad():
        acc_all = []
        for batch in tqdm.tqdm(test_dataloader):
            imgs, labels = batch
            preds = model(imgs)
            acc = (preds.argmax(dim=-1) == labels).float().mean()
            print(f"acc: {acc}")
            acc_all.append(acc.item())
        print(f"{acc_all}")
        print(f"test acc: {mean(acc_all)}")