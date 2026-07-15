import os
import sys
import torch

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.embedding_network import EmbeddingNet

# -------------------------------------------------
# Configuration
# -------------------------------------------------

MODEL_PATH = "/content/drive/MyDrive/ProductMatchingModels/triplet_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing Device : {DEVICE}")

# -------------------------------------------------
# Load Model
# -------------------------------------------------

model = EmbeddingNet(
    embedding_dim=128,
    pretrained=False
).to(DEVICE)

print("\nLoading trained weights...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("\nModel loaded successfully!")

print("\nReady for inference.")