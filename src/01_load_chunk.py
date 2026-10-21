from pathlib import Path
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = Path("data")


def load_pdf(path: Path):
    doc = pymupdf.open(path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text()

        if text.strip():
            pages.append({
                "text": text,
                "page": i + 1,
                "source": path.name
            })

    return pages


# 1. data 폴더의 모든 PDF 로딩
pdf_files = sorted(DATA_DIR.glob("*.pdf"))

print(f"PDF 파일 수: {len(pdf_files)}")

pages = []

for pdf_path in pdf_files:
    pdf_pages = load_pdf(pdf_path)
    pages.extend(pdf_pages)

    print(
        f"[로드 완료] {pdf_path.name}"
        f" -> {len(pdf_pages)} pages"
    )


print("\n==============================")
print(f"전체 텍스트 페이지 수: {len(pages)}")
print("==============================")


# 2. Recursive Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)

chunks = []

for page in pages:
    for split in splitter.split_text(page["text"]):

        if split.strip():
            chunks.append({
                "text": split,
                "page": page["page"],
                "source": page["source"]
            })


print(f"\n전체 청크 수: {len(chunks)}")


# 샘플 확인
print("\n===== 청크 샘플 =====")

for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- chunk {i} ---")
    print(f"source : {chunk['source']}")
    print(f"page   : {chunk['page']}")
    print(f"length : {len(chunk['text'])}")
    print(chunk["text"][:300])

# 3. 다음 단계에서 사용할 수 있도록 청크 저장
import json

output_path = Path("artifacts/chunks.jsonl")

with output_path.open("w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print(f"\n저장 완료: {output_path}")
