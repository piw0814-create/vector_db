import json
import time
import numpy as np
from FlagEmbedding import BGEM3FlagModel


# 1. 청크 로드
with open("artifacts/chunks.jsonl", "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

texts = [c["text"] for c in chunks]

print(f"전체 청크 수: {len(chunks)}")


# 2. BGE-M3 로드
print("BGE-M3 모델 로딩...")
model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True
)

print("모델 로딩 완료")


# 3. 전체 임베딩
print("\n전체 임베딩 시작...")
start = time.time()

output = model.encode(
    texts,
    batch_size=16,
    return_dense=True
)

embeddings = np.array(
    output["dense_vecs"]
).astype("float32")

elapsed = time.time() - start


# 4. 결과 확인
print(f"\nembedding shape : {embeddings.shape}")
print(f"dtype           : {embeddings.dtype}")
print(f"소요 시간       : {elapsed:.2f} sec")


# 5. 저장
np.save(
    "artifacts/embeddings.npy",
    embeddings
)

print("\n저장 완료: artifacts/embeddings.npy")
