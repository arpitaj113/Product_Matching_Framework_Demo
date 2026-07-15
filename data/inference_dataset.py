import os
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset


class SOPInferenceDataset(Dataset):

    def __init__(self,
                 dataset_root,
                 annotation_file,
                 transform=None):

        self.dataset_root = dataset_root
        self.transform = transform

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

        self.samples = []

        for _, row in df.iterrows():

            image_path = os.path.join(
                dataset_root,
                row["path"]
            )

            self.samples.append(
                (
                    image_path,
                    int(row["class_id"])
                )
            )

        # ---------------------------------------------------------
        # TEMPORARY: Use only first 5000 images for demo
        # ---------------------------------------------------------
        MAX_IMAGES = 5000

        if MAX_IMAGES is not None:
            self.samples = self.samples[:MAX_IMAGES]

        print(f"\nUsing {len(self.samples)} images for inference.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        image_path, class_id = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_path, class_id