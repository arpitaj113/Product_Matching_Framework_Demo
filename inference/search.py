import os
import sys
import pickle

import faiss
import numpy as np
import torch
from PIL import Image

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

from models.embedding_network import EmbeddingNet
from data.transforms import test_transform

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "triplet_model.pth"
)

INDEX_PATH = os.path.join(
    PROJECT_ROOT,
    "embeddings",
    "merged",
    "faiss.index"
)

IMAGE_PATHS = os.path.join(
    PROJECT_ROOT,
    "embeddings",
    "merged",
    "image_paths.pkl"
)

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
# Load Model
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

# ---------------------------------------------------------
# Load FAISS Index
# ---------------------------------------------------------

print("Loading FAISS Index...")

index = faiss.read_index(INDEX_PATH)

print("FAISS Loaded.")

# ---------------------------------------------------------
# Load Image Paths
# ---------------------------------------------------------

print("Loading Image Paths...")

with open(IMAGE_PATHS, "rb") as f:
    image_paths = pickle.load(f)

print("Image Paths Loaded.")
print(f"Total Indexed Images : {len(image_paths)}")

# ---------------------------------------------------------
# Generate Query Embedding
# ---------------------------------------------------------

def get_embedding(image_path):

    image = Image.open(image_path).convert("RGB")

    image = test_transform(image)

    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = model(image)

    embedding = embedding.cpu().numpy().astype(np.float32)

    # Normalize for cosine similarity
    faiss.normalize_L2(embedding)

    return embedding

# ---------------------------------------------------------
# Search Similar Images
# ---------------------------------------------------------

def search_image(query_image, top_k=5):

    if not os.path.exists(query_image):
        raise FileNotFoundError(f"Image not found:\n{query_image}")

    embedding = get_embedding(query_image)

    # Search more candidates so duplicates can be removed
    scores, indices = index.search(
        embedding,
        top_k * 3
    )

    results = []

    seen_images = set()

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        relative_path = image_paths[idx]
        image_path = os.path.join(PROJECT_ROOT,relative_path)
        image_name = os.path.basename(image_path)


        # Skip duplicate filenames
        if image_name in seen_images:
            continue

        seen_images.add(image_name)

        product_id = os.path.splitext(image_name)[0]

        results.append(
            {
                "image_path": image_path,
                "image_name": image_name,
                "product_id": product_id,
                "score": float(score)
            }
        )

        if len(results) == top_k:
            break

    return results

# ---------------------------------------------------------
# Command Line Testing
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\nProduct Matching Framework")
    print("-" * 60)

    query = input("Enter Image Path : ").strip()

    if not os.path.exists(query):

        print("\nImage not found.")

        sys.exit()

    results = search_image(
        query,
        top_k=5
    )

    print("\nTop Matches\n")

    for i, result in enumerate(results, start=1):

        print(f"Match #{i}")

        print("Image Name :", result["image_name"])

        print("Product ID :", result["product_id"])

        print(
            "Similarity :",
            f"{result['score'] * 100:.2f}%"
        )

        print("Image Path :", result["image_path"])

        print("-" * 60)