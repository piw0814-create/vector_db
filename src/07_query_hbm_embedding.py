import numpy as np
from FlagEmbedding import BGEM3FlagModel

query = "HBM이란 무엇인가?"

print("BGE-M3 로딩...")
model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True
)

print(f"질문 임베딩: {query}")

q_vec = model.encode(
    [query],
    return_dense=True
)["dense_vecs"].astype("float32")

np.save(
    "artifacts/query_hbm.npy",
    q_vec
)

print(f"shape: {q_vec.shape}")
print("저장 완료: artifacts/query_hbm.npy")
