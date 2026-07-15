"""
build_faiss.py
---------------------------------------------------------
Build a FAISS Index from merged embeddings.

Input
-----
embeddings/merged/embeddings.npy

Output
------
embeddings/merged/faiss.index
---------------------------------------------------------
"""

import os
import numpy as np
import faiss
# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

EMBEDDING_FILE = os.path.join(
    PROJECT_ROOT,
    "embeddings",
    "merged",
    "embeddings.npy"
)

INDEX_FILE = os.path.join(
    PROJECT_ROOT,
    "embeddings",
    "merged",
    "faiss.index"
)
# ---------------------------------------------------------
# Verify Embedding File
# ---------------------------------------------------------

if not os.path.exists(EMBEDDING_FILE):

    raise FileNotFoundError(
        f"Embedding file not found:\n{EMBEDDING_FILE}"
    )

print("="*60)
print("Embedding file found.")
print("="*60)

# ---------------------------------------------------------
# Load Embeddings
# ---------------------------------------------------------

print("Loading embeddings...")

embeddings = np.load(
    EMBEDDING_FILE
)

print("Embeddings Loaded.")

print()

print("Shape :", embeddings.shape)

print("Datatype :", embeddings.dtype)

# ---------------------------------------------------------
# Normalize Embeddings
# ---------------------------------------------------------

print()

print("Normalizing embeddings...")

faiss.normalize_L2(
    embeddings
)

print("Normalization Completed.")

# ---------------------------------------------------------
# Build Index
# ---------------------------------------------------------

dimension = embeddings.shape[1]

print()

print("Creating FAISS Index...")

index = faiss.IndexFlatIP(
    dimension
)

print("FAISS Index Created.")

# ---------------------------------------------------------
# Add Embeddings
# ---------------------------------------------------------

print()

print("Adding vectors to FAISS...")

index.add(
    embeddings
)

print("Vectors Added.")

# ---------------------------------------------------------
# Save Index
# ---------------------------------------------------------

faiss.write_index(
    index,
    INDEX_FILE
)

print()

print("FAISS Index Saved.")

# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------

print()

print("="*60)

print("Total Vectors :", index.ntotal)

print("Embedding Dimension :", dimension)

print("="*60)

print()

print("Everything looks good!")