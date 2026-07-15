import os
import faiss
import numpy as np

# --------------------------------------------------
# Paths
# --------------------------------------------------
EMBEDDINGS_PATH = "embeddings/embeddings.npy"
INDEX_PATH = "embeddings/faiss.index"

# --------------------------------------------------
# Check embeddings
# --------------------------------------------------
if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError(f"{EMBEDDINGS_PATH} not found!")

print("=" * 50)
print("Loading embeddings...")
print("=" * 50)

embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

print("Embedding Shape :", embeddings.shape)

dimension = embeddings.shape[1]

# --------------------------------------------------
# Build FAISS Index
# --------------------------------------------------
index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Total Indexed Images :", index.ntotal)

# --------------------------------------------------
# Save Index
# --------------------------------------------------
faiss.write_index(index, INDEX_PATH)

print("\nFAISS Index Saved Successfully!")
print("Saved at :", INDEX_PATH)