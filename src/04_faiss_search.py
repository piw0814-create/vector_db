import json
import numpy as np
import faiss


# 청크 로드
with open("artifacts/chunks.jsonl", "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

# 문서 임베딩 로드
embeddings = np.load(
    "artifacts/embeddings.npy"
).astype("float32")

# 질문 임베딩 로드
q_vec = np.load(
    "artifacts/query_embedding.npy"
).astype("float32")

with open("artifacts/query.txt", "r", encoding="utf-8") as f:
    query = f.read().strip()


print(f"문서 embedding : {embeddings.shape}")
print(f"질문 embedding : {q_vec.shape}")


# HNSW 인덱스 생성
index = faiss.IndexHNSWFlat(
    1024,
    32
)

index.hnsw.efConstruction = 200
index.add(embeddings)

print(f"FAISS 저장 벡터 수: {index.ntotal}")


# 인덱스 저장
faiss.write_index(
    index,
    "artifacts/faiss_hnsw.index"
)


# Top-5 검색
dists, idxs = index.search(
    q_vec,
    k=5
)


print("\n================================")
print(f"QUERY: {query}")
print("================================")


for rank, idx in enumerate(idxs[0], start=1):
    idx = int(idx)
    chunk = chunks[idx]

    print(f"\n[{rank}위]")
    print(f"index    : {idx}")
    print(f"distance : {dists[0][rank-1]:.4f}")
    print(f"source   : {chunk['source']}")
    print(f"page     : {chunk['page']}")
    print("text:")
    print(chunk["text"][:500])
