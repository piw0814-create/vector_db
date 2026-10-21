import json
import numpy as np
import faiss
import bm25s
from bm25_utils import create_bm25_tokenizer


QUERY = "HBM이란 무엇인가?"
TOP_K = 5


# ---------------------------------
# 데이터 로드
# ---------------------------------
with open(
    "artifacts/chunks.jsonl",
    "r",
    encoding="utf-8"
) as f:
    chunks = [json.loads(line) for line in f]

texts = [c["text"] for c in chunks]


# =================================
# 1. Dense Search
# =================================

index = faiss.read_index(
    "artifacts/faiss_hnsw.index"
)

q_vec = np.load(
    "artifacts/query_hbm.npy"
).astype("float32")

dense_distances, dense_ids = index.search(
    q_vec,
    TOP_K
)


print("\n================================")
print(f"DENSE SEARCH: {QUERY}")
print("================================")

for rank, idx in enumerate(
    dense_ids[0],
    start=1
):
    idx = int(idx)
    chunk = chunks[idx]

    print(f"\n[{rank}위]")
    print(f"id       : {idx}")
    print(
        f"distance : "
        f"{dense_distances[0][rank-1]:.4f}"
    )
    print(f"source   : {chunk['source']}")
    print(f"page     : {chunk['page']}")
    print(chunk["text"][:350])


# =================================
# 2. BM25 Search
# =================================

print("\nBM25 인덱스 생성 중...")

tokenizer = create_bm25_tokenizer()

corpus_tokens = tokenizer.tokenize(
    texts,
    update_vocab=True
)

bm25_index = bm25s.BM25()
bm25_index.index(
    corpus_tokens
)

q_tok = tokenizer.tokenize(
    [QUERY],
    update_vocab=False
)

bm25_ids, bm25_scores = bm25_index.retrieve(
    q_tok,
    k=TOP_K
)

bm25_ids = np.asarray(
    bm25_ids
).reshape(-1)

bm25_scores = np.asarray(
    bm25_scores
).reshape(-1)


print("\n================================")
print(f"BM25 SEARCH: {QUERY}")
print("================================")

for rank, idx in enumerate(
    bm25_ids,
    start=1
):
    idx = int(idx)
    chunk = chunks[idx]

    print(f"\n[{rank}위]")
    print(f"id     : {idx}")
    print(
        f"score  : "
        f"{bm25_scores[rank-1]:.4f}"
    )
    print(f"source : {chunk['source']}")
    print(f"page   : {chunk['page']}")
    print(chunk["text"][:350])
