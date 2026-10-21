import csv
import json
import numpy as np
import faiss


TOP_K = 5


# 청크
with open(
    "artifacts/chunks.jsonl",
    "r",
    encoding="utf-8"
) as f:
    chunks = [json.loads(line) for line in f]


# -----------------------------
# Dense Top-5
# -----------------------------
index = faiss.read_index(
    "artifacts/faiss_hnsw.index"
)

q_vec = np.load(
    "artifacts/query_hbm.npy"
).astype("float32")

_, dense_ids = index.search(
    q_vec,
    TOP_K
)

dense_ids = [
    int(x)
    for x in dense_ids[0]
]


# -----------------------------
# Hybrid RRF
# -----------------------------
with open(
    "artifacts/rrf_candidates.json",
    "r",
    encoding="utf-8"
) as f:
    rrf = json.load(f)

rrf_ids = rrf["candidate_ids"][:TOP_K]


# -----------------------------
# Reranker
# -----------------------------
with open(
    "artifacts/reranked_results.json",
    "r",
    encoding="utf-8"
) as f:
    reranked = json.load(f)

reranked_ids = reranked["reranked_ids"][:TOP_K]


def short(doc_id):
    c = chunks[doc_id]

    text = (
        c["text"]
        .replace("\n", " ")
        .strip()
    )

    return (
        doc_id,
        c["source"],
        c["page"],
        text[:100]
    )


print("\n============================================================")
print("Dense → Hybrid(RRF) → Reranker 비교")
print("Query: HBM이란 무엇인가?")
print("============================================================")

for rank in range(TOP_K):

    d = short(dense_ids[rank])
    h = short(rrf_ids[rank])
    r = short(reranked_ids[rank])

    print(f"\n[{rank + 1}위]")

    print(
        f"Dense    : id={d[0]}, "
        f"page={d[2]} | {d[3]}"
    )

    print(
        f"Hybrid   : id={h[0]}, "
        f"page={h[2]} | {h[3]}"
    )

    print(
        f"Reranker : id={r[0]}, "
        f"page={r[2]} | {r[3]}"
    )


# CSV 저장
with open(
    "results/three_stage_comparison.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "rank",
        "dense_id",
        "dense_page",
        "hybrid_id",
        "hybrid_page",
        "reranker_id",
        "reranker_page"
    ])

    for rank in range(TOP_K):

        d = chunks[dense_ids[rank]]
        h = chunks[rrf_ids[rank]]
        r = chunks[reranked_ids[rank]]

        writer.writerow([
            rank + 1,
            dense_ids[rank],
            d["page"],
            rrf_ids[rank],
            h["page"],
            reranked_ids[rank],
            r["page"]
        ])


print(
    "\n저장 완료: "
    "results/three_stage_comparison.csv"
)
