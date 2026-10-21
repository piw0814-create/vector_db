import json
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)


COLLECTION_NAME = "rag_docs"


# 1. 청크 로드
with open("artifacts/chunks.jsonl", "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

# 2. 임베딩 로드
embeddings = np.load(
    "artifacts/embeddings.npy"
).astype("float32")

print(f"청크 수      : {len(chunks)}")
print(f"임베딩 shape : {embeddings.shape}")


# 3. Qdrant 연결
client = QdrantClient(
    host="localhost",
    port=6333
)

print("Qdrant 연결 완료")


# 4. 기존 컬렉션이 있으면 삭제
existing = [
    c.name
    for c in client.get_collections().collections
]

if COLLECTION_NAME in existing:
    print(f"기존 {COLLECTION_NAME} 컬렉션 삭제")
    client.delete_collection(
        collection_name=COLLECTION_NAME
    )


# 5. Collection 생성
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=1024,
        distance=Distance.COSINE
    )
)

print(f"컬렉션 생성 완료: {COLLECTION_NAME}")


# 6. Point 생성
points = []

for i, chunk in enumerate(chunks):
    points.append(
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "text": chunk["text"],
                "page": chunk["page"],
                "source": chunk["source"]
            }
        )
    )


# 7. Batch 단위로 Upsert
batch_size = 200

for start in range(0, len(points), batch_size):
    batch = points[start:start + batch_size]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch
    )

    print(
        f"upsert 완료: "
        f"{start + len(batch)} / {len(points)}"
    )


# 8. Collection 정보 확인
info = client.get_collection(
    collection_name=COLLECTION_NAME
)

print("\n==============================")
print("Qdrant 적재 완료")
print("==============================")
print(f"points_count : {info.points_count}")
print(f"vectors_count: {info.indexed_vectors_count}")
