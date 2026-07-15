import os
import sys
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------
# Add Project Root
# ---------------------------------------------------------

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from models.embedding_network import EmbeddingNet
from data.inference_dataset import SOPInferenceDataset
from data.transforms import inference_transform

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET_ROOT = "/content/drive/MyDrive/Stanford_Online_Products"

ANNOTATION_FILE = os.path.join(
    DATASET_ROOT,
    "Ebay_train.txt"
)

MODEL_PATH = "/content/drive/MyDrive/ProductMatchingModels/triplet_model.pth"

SAVE_DIR = "embeddings"

EMBEDDING_DIM = 128

BATCH_SIZE = 64

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Generating Embeddings")
print("=" * 60)

print(f"Device           : {DEVICE}")
print(f"Dataset          : {DATASET_ROOT}")
print(f"Annotation File  : {ANNOTATION_FILE}")
print(f"Model            : {MODEL_PATH}")

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

dataset = SOPInferenceDataset(
    dataset_root=DATASET_ROOT,
    annotation_file=ANNOTATION_FILE,
    transform=inference_transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

print(f"\nTotal Images : {len(dataset)}")

# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------

model = EmbeddingNet(
    embedding_dim=EMBEDDING_DIM,
    pretrained=False
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("\nModel Loaded Successfully!")

# ---------------------------------------------------------
# Generate Embeddings
# ---------------------------------------------------------

embeddings = []
image_paths = []
labels = []

print("\nGenerating embeddings...\n")

with torch.no_grad():

    for images, paths, class_ids in tqdm(loader):

        images = images.to(DEVICE)

        features = model(images)

        embeddings.append(
            features.cpu().numpy()
        )

        image_paths.extend(paths)

        labels.extend(class_ids.tolist())

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

embeddings = np.vstack(embeddings)

np.save(
    os.path.join(SAVE_DIR, "embeddings.npy"),
    embeddings
)

with open(
    os.path.join(SAVE_DIR, "image_paths.pkl"),
    "wb"
) as f:
    pickle.dump(image_paths, f)

with open(
    os.path.join(SAVE_DIR, "labels.pkl"),
    "wb"
) as f:
    pickle.dump(labels, f)

print("\n" + "=" * 60)
print("Embedding Generation Completed Successfully!")
print("=" * 60)

print(f"Embeddings Shape : {embeddings.shape}")
print(f"Saved Directory  : {SAVE_DIR}")

print("\nGenerated Files:")

print("✔ embeddings.npy")
print("✔ image_paths.pkl")
print("✔ labels.pkl")