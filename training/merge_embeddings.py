"""
merge_embeddings.py
---------------------------------------------------------
Merge all embedding batches and image path batches
into single files.

Output:
    embeddings/merged/embeddings.npy
    embeddings/merged/image_paths.pkl
---------------------------------------------------------
"""

import os
import pickle
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

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

MERGED_DIR = os.path.join(
    PROJECT_ROOT,
    "embeddings",
    "merged"
)

os.makedirs(MERGED_DIR, exist_ok=True)

# ---------------------------------------------------------
# Get Batch Files
# ---------------------------------------------------------

embedding_files = sorted([
    f for f in os.listdir(EMBED_BATCH_DIR)
    if f.endswith(".npy")
])

path_files = sorted([
    f for f in os.listdir(PATH_BATCH_DIR)
    if f.endswith(".pkl")
])

print("=" * 60)
print(f"Embedding Batch Files : {len(embedding_files)}")
print(f"Path Batch Files      : {len(path_files)}")
print("=" * 60)

# ---------------------------------------------------------
# Merge Embeddings
# ---------------------------------------------------------

all_embeddings = []

print("\nMerging Embeddings...\n")

for file in tqdm(embedding_files):

    embedding = np.load(
        os.path.join(
            EMBED_BATCH_DIR,
            file
        )
    )

    all_embeddings.append(embedding)

all_embeddings = np.vstack(all_embeddings)

print("\nFinal Embedding Shape :", all_embeddings.shape)

# ---------------------------------------------------------
# Merge Image Paths
# ---------------------------------------------------------

all_paths = []

print("\nMerging Image Paths...\n")

for file in tqdm(path_files):

    with open(
        os.path.join(
            PATH_BATCH_DIR,
            file
        ),
        "rb"
    ) as f:

        paths = pickle.load(f)

    all_paths.extend(paths)

print("Total Image Paths :", len(all_paths))

# ---------------------------------------------------------
# Save Merged Files
# ---------------------------------------------------------

embedding_output = os.path.join(
    MERGED_DIR,
    "embeddings.npy"
)

path_output = os.path.join(
    MERGED_DIR,
    "image_paths.pkl"
)

np.save(
    embedding_output,
    all_embeddings
)

with open(path_output, "wb") as f:
    pickle.dump(all_paths, f)

print("\nMerged files saved successfully.")

# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------

print("\nVerification")

print("-" * 50)

print("Embeddings Shape :", all_embeddings.shape)

print("Image Paths      :", len(all_paths))

assert all_embeddings.shape[0] == len(all_paths)

print("\nEverything looks correct!")