from vectorize.text_splitters import CodeTextSplitter

def process_code_file(file_path):
    with open(file_path, encoding="utf-8") as f:
        text = f.read()
    splitter = CodeTextSplitter()
    chunks = splitter.split(text)
    return [
        {
            "content": chunk,
            "metadata": {
                "source": file_path,
                "source_type": "code"
            }
        }
        for chunk in chunks
    ]