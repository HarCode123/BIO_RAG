import os
import pickle
import numpy as np
import faiss

# Load embeddings
with open("vectorstore/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Convert to NumPy float32
embeddings = np.array(embeddings, dtype="float32")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings
index.add(embeddings)

# Save the index
faiss.write_index(index, "vectorstore/index.faiss")

print("FAISS index created successfully!")
print("Total vectors:", index.ntotal)
print("Embedding dimension:", dimension)