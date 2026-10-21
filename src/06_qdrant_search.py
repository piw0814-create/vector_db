import json
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    Range,
)


COLLECTION_NAME = "rag_docs"


# 질문 벡터 로드
q_vec = np.load(
    "artifacts/query_embedding.npy"
).astype("float32")

with open(
    "artifacts/query.txt",
    "r",
    encoding="utf-8"
) as f:
    query = f.read().strip()


client = QdrantClient(
    host="localhost",
    port=6333
)


# -----------------------------
# 1. 일반 검색
# -----------------------------
results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=q_vec[0].tolist(),
    limit=5,
    with_payload=True,
).points


print("\n================================")
print(f"QDRANT SEARCH: {query}")
print("================================")

for rank, result in enumerate(results, start=1):
    payload = result.payload

    print(f"\n[{rank}위]")
    print(f"score  : {result.score:.4f}")
    print(f"source : {payload['source']}")
    print(f"page   : {payload['page']}")
    print("text:")
    print(payload["text"][:500])


# -----------------------------
# 2. Metadata Filter 검색
#    page >= 5
# -----------------------------
filtered = client.query_points(
    collection_name=COLLECTION_NAME,
    query=q_vec[0].tolist(),

    query_filter=Filter(
        must=[
            FieldCondition(
                key="page",
                range=Range(gte=5)
            )
        ]
    ),

    limit=5,
    with_payload=True,
).points


print("\n\n================================")
print("FILTER SEARCH: page >= 5")
print("================================")

for rank, result in enumerate(filtered, start=1):
    payload = result.payload

    print(f"\n[{rank}위]")
    print(f"score  : {result.score:.4f}")
    print(f"source : {payload['source']}")
    print(f"page   : {payload['page']}")
    print("text:")
    print(payload["text"][:500])
