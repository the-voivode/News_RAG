# faiss_playground.py

documents = [
    "The United States escalated its involvement in Vietnam in 1965.",
    "Elysium is a vast world, stitched together from isolae and pale.",
    "Creatine supplementation increases phosphocreatine stores in muscles.",
    "Tehran has severe air pollution, especially during winter inversions.",
    "Retrieval-augmented generation uses external knowledge sources with LLMs.",
    "Epicurus is the most misunderstood philosopher in the world.",
    "Valve's announcements are pointing to the new half life game.",
    "It's better to die a hero rather to live knowing that you're a monster",
    "The initiate must complete one last rite, he won't be able to, unless he's understood all he's achieved before.",
    "When the shift ends, the cosplay ends with it."
]


from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-large-en"
model = SentenceTransformer(MODEL_NAME)

def embed_text(docs):
    embeddings = model.encode(
        docs,
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=True
    )
    return embeddings


import faiss
import numpy as np

def build_faiss_index(embeddings):
    embeddings = embeddings.astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def search(query, index, embeddings, docs, k=3):
    q_emb = embed_text([query])[0]        # shape (1024,)
    q_emb = np.array([q_emb], dtype="float32")   # shape (1, 1024)

    distances, indeces = index.search(q_emb, k)

    results = []

    for dist, idx in zip(distances[0], indeces[0]):
        results.append({
            "doc": documents[idx],
            "distance": float(dist),
            "index": int(idx)
        })
    return results




if __name__ == "__main__":
    embs = embed_text(documents)
    # print("embedding size:", embs.shape)

    index = build_faiss_index(embs)
    # print("number of vectors in index:", index.ntotal)

    while True:
        query = input("\n enter query. (enter q to exit)").strip()
        if query == "q":
            break
        
        results = search(query, index, embs, documents, k=3)
        for r in results:
            print(f"distance: [{r['distance']:.4f}]", f"doc: {r['doc']}")