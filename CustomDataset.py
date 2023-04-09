import torch
import pandas as pd
from PIL import Image


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transforms):
        super().__init__()
        self.df = pd.read_parquet(dataset_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        local_url = self.df.iloc[idx]["LOCAL_PATH"]
        target = self.df.iloc[idx]["TARGET"]
        image = self.transforms(Image.open(local_url).convert("RGB"))
        return image, target
