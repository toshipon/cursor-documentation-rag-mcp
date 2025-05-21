import pdfplumber
from vectorize.text_splitters import BaseTextSplitter

def process_pdf_file(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    splitter = BaseTextSplitter()
    chunks = splitter.split(text)
    return [
        {
            "content": chunk,
            "metadata": {
                "source": file_path,
                "source_type": "pdf"
            }
        }
        for chunk in chunks
    ]