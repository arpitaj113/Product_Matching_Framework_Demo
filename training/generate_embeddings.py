"""
generate_embeddings.py
---------------------------------------------------------
Generate embeddings for the Stanford Online Products dataset
using the trained Triplet Network.

Features
--------
- CPU/GPU compatible
- Batch processing
- Resume support (will be added in Part 3)
- Saves embeddings batch-wise
---------------------------------------------------------
"""
import pickle
import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------
# Add Project Root to Python Path
# ---------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(PROJECT_ROOT)

# ---------------------------------------------------------
# Import Project Modules
# ---------------------------------------------------------

from models.embedding_network import EmbeddingNet
from data.transforms import test_transform

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "triplet_model.pth"
)

BATCH_FOLDER = os.path.join(
    PROJECT_ROOT,
    "embeddings",
    "batches"
)

MERGED_FOLDER = os.path.join(
    PROJECT_ROOT,
    "embeddings",
    "merged"
)

LOG_FOLDER = os.path.join(
    PROJECT_ROOT,
    "embeddings",
    "logs"
)

# ---------------------------------------------------------
# Create Required Folders
# ---------------------------------------------------------

os.makedirs(BATCH_FOLDER, exist_ok=True)
os.makedirs(MERGED_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Device :", DEVICE)
print("=" * 60)

# ---------------------------------------------------------
# Load Trained Model
# ---------------------------------------------------------

print("Loading Triplet Model...")

model = EmbeddingNet(
    embedding_dim=128,
    pretrained=False
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model Loaded Successfully.")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

print("Loading Triplet Model...")

model = EmbeddingNet(
    embedding_dim=128,
    pretrained=False
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)
# ---------------------------------------------------------
# Verify Dataset
# ---------------------------------------------------------

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset folder not found:\n{DATASET_PATH}"
    )

print("Dataset Found.")

# ---------------------------------------------------------
# Collect Image Paths
# ---------------------------------------------------------

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)

all_images = []

for root, _, files in os.walk(DATASET_PATH):

    for file in files:

        if file.lower().endswith(IMAGE_EXTENSIONS):

            all_images.append(
                os.path.join(root, file)
            )

print(f"Total Images Found : {len(all_images)}")

# ---------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------

class ProductDataset(Dataset):

    def __init__(self, image_paths, transform=None):

        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        try:

            image = Image.open(image_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            relative_path = os.path.relpath(image_path, PROJECT_ROOT)
            relative_path = relative_path.replace(os.sep, "/")
            return image, relative_path

        except Exception as e:

            print(f"Error loading image:\n{image_path}")
            print(e)

            return None, image_path

# ---------------------------------------------------------
# Custom Collate Function
# ---------------------------------------------------------

def collate_fn(batch):

    batch = [item for item in batch if item[0] is not None]

    if len(batch) == 0:
        return None, None

    images, paths = zip(*batch)

    images = torch.stack(images)

    return images, paths

# ---------------------------------------------------------
# DataLoader Configuration
# ---------------------------------------------------------

BATCH_SIZE = 64

# ---------------------------------------------------------
# Create Dataset
# ---------------------------------------------------------

dataset = ProductDataset(
    image_paths=all_images,
    transform=test_transform
)

print("Dataset Created Successfully.")
# ---------------------------------------------------------
# Create DataLoader
# ---------------------------------------------------------

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn
)

print("DataLoader Created Successfully.")

# ---------------------------------------------------------
# Dataset Information
# ---------------------------------------------------------

print("=" * 60)
print("Total Images :", len(dataset))
print("Batch Size   :", BATCH_SIZE)
print("Total Batches:", len(loader))
print("=" * 60)
# ---------------------------------------------------------
# Test DataLoader
# ---------------------------------------------------------

print("\nTesting DataLoader...\n")

print("\nTesting DataLoader...\n")

images, paths = next(iter(loader))

if images is None:
    print("No valid images found in the first batch.")
else:
    print("Image Batch Shape :", images.shape)
    print("First Image Path :")
    print(paths[0])

# ---------------------------------------------------------
# Batch Saving Paths
# ---------------------------------------------------------

EMBED_BATCH_DIR = os.path.join(
    PROJECT_ROOT,
    "embeddings",
    "batches",
    "embeddings"
)

PATH_BATCH_DIR = os.path.join(
    PROJECT_ROOT,
    "embeddings",
    "batches",
    "paths"
)

os.makedirs(EMBED_BATCH_DIR, exist_ok=True)
os.makedirs(PATH_BATCH_DIR, exist_ok=True)

# ---------------------------------------------------------
# Generate Embeddings Batch-wise
# ---------------------------------------------------------

print("\nStarting Embedding Generation...\n")

model.eval()
for batch_idx, batch in enumerate(tqdm(loader)):

    images, paths = batch

    if images is None:
        continue

    embed_file = os.path.join(
        EMBED_BATCH_DIR,
        f"batch_{batch_idx:05d}.npy"
    )

    path_file = os.path.join(
        PATH_BATCH_DIR,
        f"batch_{batch_idx:05d}.pkl"
    )

    # Skip if already processed
    if os.path.exists(embed_file) and os.path.exists(path_file):
        print(f"Skipping Batch {batch_idx + 1} (already exists)")
        continue

    images = images.to(DEVICE)

    with torch.no_grad():

        embeddings = model(images)

        embeddings = embeddings.cpu().numpy().astype(np.float32)

    np.save(
        embed_file,
        embeddings
    )

    with open(path_file, "wb") as f:
        pickle.dump(paths, f)

print("\nEmbedding Generation Completed Successfully.")