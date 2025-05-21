import argparse
import os
from vectorize.embeddings import DummyEmbedder
from db.vector_store import VectorStore
from vectorize.processors.markdown_processor import process_markdown_file
from vectorize.processors.pdf_processor import process_pdf_file
from vectorize.processors.code_processor import process_code_file

def get_processor(file_path):
    if file_path.endswith(".md"):
        return process_markdown_file
    elif file_path.endswith(".pdf"):
        return process_pdf_file
    elif file_path.endswith(".py"):
        return process_code_file
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description="ドキュメントをベクター化してDBに格納します")
    parser.add_argument("--input_dir", required=True, help="ドキュメントディレクトリ")
    parser.add_argument("--output_db", required=True, help="出力DBパス")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_db = args.output_db

    embedder = DummyEmbedder()
    vector_store = VectorStore(output_db)

    for root, _, files in os.walk(input_dir):
        for fname in files:
            file_path = os.path.join(root, fname)
            processor = get_processor(file_path)
            if processor is None:
                print(f"スキップ: {file_path}")
                continue
            docs = processor(file_path)
            vectors = embedder.embed_batch([d["content"] for d in docs])
            vector_store.add_documents(docs, vectors, file_path=file_path)
            print(f"格納完了: {file_path}")

if __name__ == "__main__":
    main()