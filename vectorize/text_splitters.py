import re

class BaseTextSplitter:
    def __init__(self, chunk_size=200):
        self.chunk_size = chunk_size

    def split(self, text):
        # 指定トークン数ごとに分割（ここでは単純に文字数で分割）
        return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

class MarkdownTextSplitter(BaseTextSplitter):
    def split(self, text):
        # 見出しごとに分割し、さらに大きい場合はchunk_sizeで分割
        sections = re.split(r'(^#+ .+$)', text, flags=re.MULTILINE)
        chunks = []
        buf = ""
        for part in sections:
            if part.startswith("#"):
                if buf:
                    chunks.extend(super().split(buf))
                    buf = ""
                buf += part + "\n"
            else:
                buf += part
        if buf:
            chunks.extend(super().split(buf))
        return chunks

class CodeTextSplitter(BaseTextSplitter):
    def split(self, text):
        # 関数やクラス定義ごとに分割（Python例）
        blocks = re.split(r'(^def |^class )', text, flags=re.MULTILINE)
        chunks = []
        buf = ""
        for part in blocks:
            if part.startswith("def ") or part.startswith("class "):
                if buf:
                    chunks.extend(super().split(buf))
                    buf = ""
                buf += part
            else:
                buf += part
        if buf:
            chunks.extend(super().split(buf))
        return chunks