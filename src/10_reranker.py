import json
from FlagEmbedding import FlagReranker


# ---------------------------------
# 데이터 로드
# ---------------------------------
with open(
    "artifacts/chunks.jsonl",
    "r",
    encoding="utf-8"
) as f:
    chunks = [json.loads(line) for line in f]


with open(
    "artifacts/rrf_candidates.json",
    "r",
    encoding="utf-8"
) as f:
    rrf_data = json.load(f)


query = rrf_data["query"]
candidate_ids = rrf_data["candidate_ids"]


print(f"Query        : {query}")
print(f"후보 문서 수 : {len(candidate_ids)}")


# ---------------------------------
# 후보 문서 준비
# ---------------------------------
candidates = [
    chunks[doc_id]["text"]
    for doc_id in candidate_ids
]

pairs = [
    [query, text]
    for text in candidates
]


# ---------------------------------
# BGE Reranker
# ---------------------------------
print("\nBGE Reranker 로딩...")

reranker = FlagReranker(
    "BAAI/bge-reranker-v2-m3",
    use_fp16=True
)

print("Reranker 재채점 시작...")


scores = reranker.compute_score(
    pairs,
    normalize=True
)


# ---------------------------------
# 점수 + ID 결합
# ---------------------------------
reranked = sorted(
    zip(
        scores,
        candidate_ids
    ),
    key=lambda x: x[0],
    reverse=True
)


# ---------------------------------
# 결과 출력
# ---------------------------------
print("\n================================")
print(f"RERANKED SEARCH: {query}")
print("================================")


for rank, (score, doc_id) in enumerate(
    reranked,
    start=1
):
    chunk = chunks[doc_id]

    original_rrf_rank = (
        candidate_ids.index(doc_id) + 1
    )

    print(f"\n[{rank}위]")
    print(f"id            : {doc_id}")
    print(f"RRF rank      : {original_rrf_rank}")
    print(f"Reranker score: {float(score):.4f}")
    print(f"source        : {chunk['source']}")
    print(f"page          : {chunk['page']}")
    print(chunk["text"][:400])


# ---------------------------------
# 다음 단계용 저장
# ---------------------------------
result = {
    "query": query,
    "reranked_ids": [
        int(doc_id)
        for score, doc_id in reranked
    ],
    "scores": [
        float(score)
        for score, doc_id in reranked
    ]
}


with open(
    "artifacts/reranked_results.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        result,
        f,
        ensure_ascii=False,
        indent=2
    )


print(
    "\n저장 완료: "
    "artifacts/reranked_results.json"
)
