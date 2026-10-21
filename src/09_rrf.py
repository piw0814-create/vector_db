import json
import numpy as np
import faiss
import bm25s
from bm25_utils import create_bm25_tokenizer


QUERY = "HBM이란 무엇인가?"
CANDIDATE_K = 15
FINAL_K = 10
RRF_K = 60


# -----------------------------
# 데이터 로드
# -----------------------------
with open("artifacts/chunks.jsonl", "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

texts = [c["text"] for c in chunks]


# -----------------------------
# 1. Dense Top-K
# -----------------------------
index = faiss.read_index(
    "artifacts/faiss_hnsw.index"
)

q_vec = np.load(
    "artifacts/query_hbm.npy"
).astype("float32")

_, dense_ids = index.search(
    q_vec,
    CANDIDATE_K
)

dense_ids = [
    int(i) for i in dense_ids[0]
]

dense_ranks = {
    doc_id: rank
    for rank, doc_id in enumerate(
        dense_ids,
        start=1
    )
}


# -----------------------------
# 2. BM25 Top-K
# -----------------------------
tokenizer = create_bm25_tokenizer()

corpus_tokens = tokenizer.tokenize(
    texts,
    update_vocab=True,
    show_progress=False
)

bm25_index = bm25s.BM25()
bm25_index.index(
    corpus_tokens,
    show_progress=False
)

query_tokens = tokenizer.tokenize(
    [QUERY],
    update_vocab=False,
    show_progress=False
)

bm25_ids, _ = bm25_index.retrieve(
    query_tokens,
    k=CANDIDATE_K
)

bm25_ids = [
    int(i)
    for i in np.asarray(bm25_ids).reshape(-1)
]

bm25_ranks = {
    doc_id: rank
    for rank, doc_id in enumerate(
        bm25_ids,
        start=1
    )
}


# -----------------------------
# 3. RRF
# -----------------------------
all_ids = (
    set(dense_ranks)
    | set(bm25_ranks)
)

rrf_scores = {}

for doc_id in all_ids:

    score = 0.0

    if doc_id in dense_ranks:
        score += 1 / (
            RRF_K + dense_ranks[doc_id]
        )

    if doc_id in bm25_ranks:
        score += 1 / (
            RRF_K + bm25_ranks[doc_id]
        )

    rrf_scores[doc_id] = score


hybrid_ids = sorted(
    rrf_scores,
    key=rrf_scores.get,
    reverse=True
)[:FINAL_K]


# -----------------------------
# 결과 출력
# -----------------------------
print("\n================================")
print(f"HYBRID RRF SEARCH: {QUERY}")
print("================================")

for rank, doc_id in enumerate(
    hybrid_ids,
    start=1
):
    chunk = chunks[doc_id]

    print(f"\n[{rank}위]")
    print(f"id         : {doc_id}")
    print(
        f"dense rank : "
        f"{dense_ranks.get(doc_id, '-')}"
    )
    print(
        f"BM25 rank  : "
        f"{bm25_ranks.get(doc_id, '-')}"
    )
    print(
        f"RRF score  : "
        f"{rrf_scores[doc_id]:.6f}"
    )
    print(f"source     : {chunk['source']}")
    print(f"page       : {chunk['page']}")
    print(chunk["text"][:350])


# -----------------------------
# Reranker에서 사용할 후보 저장
# -----------------------------
with open(
    "artifacts/rrf_candidates.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "query": QUERY,
            "candidate_ids": hybrid_ids
        },
        f,
        ensure_ascii=False,
        indent=2
    )

print(
    "\n후보 저장 완료: "
    "artifacts/rrf_candidates.json"
)
