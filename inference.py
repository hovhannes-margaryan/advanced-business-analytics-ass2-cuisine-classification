import torch
from torchvision import transforms
from argparse import ArgumentParser
from torch.utils.data import DataLoader
# from CustomDataset import CustomDataset
from torchvision.datasets import ImageFolder
from models import VisionTransformer
from modules import GeneralMLightningModule
import torch.nn.functional as F
from statistics import mean
import tqdm
from torchmetrics import Accuracy
from sklearn.metrics import confusion_matrix as confusion_matrix_skl
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


parser = ArgumentParser()
parser.add_argument("--pretrained_model_path", help="path to the pretrained model", type=str)
parser.add_argument("--test_parquet_path", help="path to the train parquet", type=str)
parser.add_argument("--device", default="cuda", help="training and validation batch size", type=str)
parser.add_argument("--batch_size", default=128, help="training and validation batch size", type=int)
parser.add_argument("--image_size", default=224, help="size of the image for training", type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device(args.device)
    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                             [0.24703223, 0.24348513, 0.26158784])])
    test_dataset = ImageFolder(args.test_parquet_path, test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 drop_last=False, num_workers=4)

    model = GeneralMLightningModule.load_from_checkpoint(args.pretrained_model_path)
    model.eval()
    accuracy = Accuracy(task="binary", num_classes=2)
    all_labels = []
    all_preds = []
    all_preds_softmax = []
    with torch.no_grad():
        acc_all = []
        for i, batch in tqdm.tqdm(enumerate(test_dataloader)):
            imgs, labels = batch
            preds = model(imgs)
            preds_softmax = F.softmax(preds, dim=1)
            acc = (preds.argmax(dim=-1) == labels).float().mean()
            acc_all.append(acc.item())
            all_labels.append(labels)
            all_preds_softmax.append(preds_softmax.argmax(dim=-1))
            all_preds.append(preds.argmax(dim=-1))
        print(f"test acc: {mean(acc_all)}")

        all_labels = [item for sublist in all_labels for item in sublist]
        all_preds = [item for sublist in all_preds for item in sublist]
        all_preds_softmax = [item for sublist in all_preds_softmax for item in sublist]
        all_labels_tensor = torch.tensor(all_labels)
        all_preds_softmax_tensor = torch.tensor(all_preds_softmax)
        all_preds_tensor = torch.tensor(all_preds)

        acc = (all_preds_tensor == all_labels_tensor).float().mean()
        acc_lightning = accuracy(all_preds_softmax_tensor, all_labels_tensor)
        all_preds_softmax = [1-x for x in all_preds_softmax]
        all_labels = [1-x for x in all_labels]
        fpr, tpr, _ = roc_curve(all_preds_softmax, all_labels)
        auc = roc_auc_score(all_preds_softmax, all_labels)

        plt.plot(fpr, tpr, label="AUC=" + str(auc))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.show()

        print(acc, acc_lightning)
        print(confusion_matrix_skl(all_preds_softmax, all_labels))