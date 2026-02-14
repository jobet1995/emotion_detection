import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms


class EmotionDataset(Dataset):
    def __init__(self, dataframe, image_size=48, augment=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_size = image_size
        self.augment = augment
        self.transform = self._build_transforms()

    def _build_transforms(self):
        if self.augment:
            return transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        return transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row["filepath"]).convert("L")
        image = self.transform(image)
        label = torch.tensor(row["label_encoded"], dtype=torch.long)
        return image, label


class EmotionDataModule:
    def __init__(
        self,
        csv_path,
        batch_size=32,
        image_size=48,
        val_split=0.2,
        random_state=42,
        num_workers=2
    ):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.val_split = val_split
        self.random_state = random_state
        self.num_workers = num_workers
        self._prepare_data()

    def _prepare_data(self):
        self.df = pd.read_csv(self.csv_path)
        train_df, val_df = train_test_split(
            self.df,
            test_size=self.val_split,
            stratify=self.df["label_encoded"],
            random_state=self.random_state
        )
        self.train_dataset = EmotionDataset(
            train_df,
            image_size=self.image_size,
            augment=True
        )
        self.val_dataset = EmotionDataset(
            val_df,
            image_size=self.image_size,
            augment=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def get_num_classes(self):
        return self.df["label_encoded"].nunique()

if __name__ == "__main__":
    data = EmotionDataModule(
        csv_path="data/processed/full_processed_dataset.csv",
        batch_size=32,
        image_size=48
    )

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()

    images, labels = next(iter(train_loader))

    print("Train batch shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Number of classes:", data.get_num_classes())

