import os
import pickle
from dotenv import load_dotenv
from google import genai
from ingest import load_chunks
import pickle
from ingest import load_chunks   # import from ingest.py

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-embedding-001"


def get_embedding(text):
    response = client.models.embed_content(
        model=MODEL,
        contents=text
    )
    return response.embeddings[0].values


def create_embeddings(chunks):
    embeddings = []

    for i, chunk in enumerate(chunks):
        print(f"Embedding {i+1}/{len(chunks)}")
        embeddings.append(get_embedding(chunk))

    return embeddings


if __name__ == "__main__":

    chunks = load_chunks()
    print(f"Total chunks: {len(chunks)}")

    embeddings = create_embeddings(chunks)

    os.makedirs("vectorstore", exist_ok=True)

    with open("vectorstore/embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    with open("vectorstore/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Done!")