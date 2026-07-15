import os
import random

import pandas as pd
from PIL import Image

from torch.utils.data import Dataset


class SOPTripletDataset(Dataset):

    def __init__(self,
                 dataset_root,
                 annotation_file,
                 transform=None):

        self.dataset_root = dataset_root
        self.transform = transform

        # Read annotation file
        df = pd.read_csv(
            annotation_file,
            sep=r"\s+",
            skiprows=1,
            names=[
                "image_id",
                "class_id",
                "super_class_id",
                "path"
            ]
        )

        self.class_to_images = {}

        for _, row in df.iterrows():

            class_id = int(row["class_id"])

            image_path = os.path.join(
                dataset_root,
                row["path"]
            )

            self.class_to_images.setdefault(
                class_id,
                []
            ).append(image_path)

        self.classes = list(self.class_to_images.keys())

        self.samples = []

        for cls in self.classes:
            for img in self.class_to_images[cls]:
                self.samples.append((img, cls))

        print(f"\nTotal Classes : {len(self.classes)}")
        print(f"Total Images  : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        anchor_path, anchor_class = self.samples[index]

        positive_path = anchor_path

        while positive_path == anchor_path:
            positive_path = random.choice(
                self.class_to_images[anchor_class]
            )

        negative_class = anchor_class

        while negative_class == anchor_class:
            negative_class = random.choice(
                self.classes
            )

        negative_path = random.choice(
            self.class_to_images[negative_class]
        )

        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        if self.transform:

            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return (
            anchor,
            positive,
            negative,
            anchor_class,
            negative_class
        )