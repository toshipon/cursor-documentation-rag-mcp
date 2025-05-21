from vectorize.text_splitters import MarkdownTextSplitter

def process_markdown_file(file_path):
    with open(file_path, encoding="utf-8") as f:
        text = f.read()
    splitter = MarkdownTextSplitter()
    chunks = splitter.split(text)
    # メタデータ例
    return [
        {
            "content": chunk,
            "metadata": {
                "source": file_path,
                "source_type": "markdown"
            }
        }
        for chunk in chunks
    ]