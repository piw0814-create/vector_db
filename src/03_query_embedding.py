import numpy as np
from FlagEmbedding import BGEM3FlagModel

query = "HBM이란 무엇인가?"

print("BGE-M3 로딩...")
model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True
)

print("질문 임베딩...")
q_vec = model.encode(
    [query],
    return_dense=True
)["dense_vecs"].astype("float32")

print(f"query embedding shape: {q_vec.shape}")

np.save("artifacts/query_embedding.npy", q_vec)

with open("artifacts/query.txt", "w", encoding="utf-8") as f:
    f.write(query)

print("저장 완료: artifacts/query_embedding.npy")
