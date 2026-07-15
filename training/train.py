import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import SOPTripletDataset
from data.transforms import train_transform
from models.embedding_network import EmbeddingNet

# -----------------------
# Configuration
# -----------------------
DATASET_ROOT = "/content/Stanford_Online_Products"

TRAIN_FILE = os.path.join(DATASET_ROOT, "Ebay_train.txt")
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 128
MODEL_SAVE_PATH = "/content/drive/MyDrive/ProductMatchingModels/triplet_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing Device: {device}")

# -----------------------
# Dataset
# -----------------------
train_dataset = SOPTripletDataset(
    dataset_root=DATASET_ROOT,
    annotation_file=TRAIN_FILE,
    transform=train_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)

# -----------------------
# Model
# -----------------------
model = EmbeddingNet(
    embedding_dim=EMBEDDING_DIM,
    pretrained=True
).to(device)

# -----------------------
# Loss
# -----------------------
criterion = nn.TripletMarginLoss(
    margin=1.0,
    p=2
)

# -----------------------
# Optimizer
# -----------------------
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

# -----------------------
# Training Loop
# -----------------------
best_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    progress = tqdm(train_loader)

    for anchor, positive, negative, _, _ in progress:

        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()

        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        negative_embedding = model(negative)

        loss = criterion(
            anchor_embedding,
            positive_embedding,
            negative_embedding
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        progress.set_description(
            f"Epoch {epoch+1}/{EPOCHS}"
        )

        progress.set_postfix(
            loss=loss.item()
        )

    epoch_loss = running_loss / len(train_loader)

    print(f"\nEpoch {epoch+1} Loss : {epoch_loss:.4f}")

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        os.makedirs("saved_models", exist_ok=True)

        torch.save(
            model.state_dict(),
            MODEL_SAVE_PATH
        )

        print("Best Model Saved!")

print("\nTraining Completed!")