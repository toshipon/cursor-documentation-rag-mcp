import re
import os
from typing import List, Dict, Any, Tuple, Optional

class BaseTextSplitter:
    """
    基本的なテキスト分割クラス
    テキストを指定サイズのチャンクに分割し、メタデータを抽出する
    """
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50):
        """
        初期化
        
        Args:
            chunk_size: 分割するチャンクのサイズ（文字数）
            chunk_overlap: チャンク間のオーバーラップ文字数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        テキストを指定サイズのチャンクに分割
        
        Args:
            text: 分割するテキスト
            
        Returns:
            分割されたテキストのリスト
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        
        while start < len(text):
            # チャンクの終了位置を計算
            end = min(start + self.chunk_size, len(text))
            
            # チャンクを追加
            chunk = text[start:end]
            chunks.append(chunk)
            
            # 次のチャンクの開始位置を計算（オーバーラップを考慮）
            start = end - self.chunk_overlap
            
            # オーバーラップが負の値になるのを防ぐ
            if start < 0:
                start = 0
                
            # 進捗がない場合は終了
            if start >= end:
                break
                
        return chunks

    def split_with_metadata(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        テキストを分割し、各チャンクにメタデータを付与
        
        Args:
            text: 分割するテキスト
            metadata: 共通メタデータ（ファイルパスなど）
            
        Returns:
            コンテンツとメタデータを含む辞書のリスト
        """
        chunks = self.split_text(text)
        
        if metadata is None:
            metadata = {}
            
        result = []
        for i, chunk in enumerate(chunks):
            # 基本メタデータをコピー
            chunk_metadata = metadata.copy()
            
            # チャンク固有のメタデータを追加
            chunk_metadata.update({
                "chunk_index": i,
                "chunk_total": len(chunks)
            })
            
            result.append({
                "content": chunk,
                "metadata": chunk_metadata
            })
            
        return result

class MarkdownTextSplitter(BaseTextSplitter):
    """
    マークダウンテキスト専用の分割クラス
    見出しやセクションの構造を考慮して分割する
    """
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        
    def extract_title(self, text: str) -> Tuple[str, Optional[str]]:
        """
        マークダウンから最初の見出しをタイトルとして抽出
        
        Args:
            text: マークダウンテキスト
            
        Returns:
            (タイトル, 残りのテキスト) のタプル
        """
        # 最初の見出し行を探す（# で始まる行）
        title_match = re.search(r'^(#+ .+)$', text, re.MULTILINE)
        
        if title_match:
            title = title_match.group(1).strip('# ')
            return title, text
        
        # 見出しがない場合はファイル名や最初の行をタイトルとする
        first_line = text.strip().split('\n')[0] if text else ""
        return first_line, text
    
    def split_text(self, text: str) -> List[str]:
        """
        マークダウンを見出し構造を考慮して分割
        
        Args:
            text: マークダウンテキスト
            
        Returns:
            分割されたチャンクのリスト
        """
        if not text:
            return []
            
        # 見出し行でテキストを分割
        sections = re.split(r'(^#{1,3} .+$)', text, flags=re.MULTILINE)
        
        chunks = []
        current_section = ""
        
        for i, section in enumerate(sections):
            # 見出し行の場合
            if i > 0 and re.match(r'^#{1,3} ', section):
                # 前のセクションが大きすぎる場合、さらに分割
                if len(current_section) > self.chunk_size:
                    sub_chunks = super().split_text(current_section)
                    chunks.extend(sub_chunks)
                elif current_section:
                    chunks.append(current_section)
                
                # 新しいセクションを開始
                current_section = section
            else:
                # 本文部分を現在のセクションに追加
                current_section += section
        
        # 最後のセクションを処理
        if len(current_section) > self.chunk_size:
            sub_chunks = super().split_text(current_section)
            chunks.extend(sub_chunks)
        elif current_section:
            chunks.append(current_section)
            
        return chunks
    
    def split_with_metadata(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        マークダウンを分割し、構造情報をメタデータに追加
        
        Args:
            text: マークダウンテキスト
            metadata: 基本メタデータ
            
        Returns:
            コンテンツとメタデータを含む辞書のリスト
        """
        if metadata is None:
            metadata = {}
            
        # タイトルを抽出
        title, _ = self.extract_title(text)
        base_metadata = metadata.copy()
        base_metadata["title"] = title
        
        # テキストを分割
        chunks = self.split_text(text)
        
        result = []
        for i, chunk in enumerate(chunks):
            # 各チャンクのメタデータを作成
            chunk_metadata = base_metadata.copy()
            
            # 見出しレベルとセクションタイトルを抽出
            heading_match = re.search(r'^(#+) (.+)$', chunk, re.MULTILINE)
            if heading_match:
                heading_level = len(heading_match.group(1))
                section_title = heading_match.group(2).strip()
                
                chunk_metadata.update({
                    "heading_level": heading_level,
                    "section_title": section_title
                })
                
            # チャンクインデックス情報
            chunk_metadata.update({
                "chunk_index": i,
                "chunk_total": len(chunks)
            })
            
            result.append({
                "content": chunk,
                "metadata": chunk_metadata
            })
            
        return result

class CodeTextSplitter(BaseTextSplitter):
    """
    ソースコード専用の分割クラス
    関数、クラス、メソッドなどの構造を考慮して分割
    """
    LANGUAGE_PATTERNS = {
        "python": {
            "class_pattern": r'(^class \w+[\(\w\)]*:)',
            "function_pattern": r'(^def \w+\([^\)]*\):)',
            "comment_pattern": r'^\s*#.*$'
        },
        "javascript": {
            "class_pattern": r'(^class \w+ {)',
            "function_pattern": r'(^(async )?function \w+\([^\)]*\) {)|(\w+ = (\(\) =>|function\([^\)]*\)) {)',
            "comment_pattern": r'^\s*(\/\/.*$|\/\*[\s\S]*?\*\/)'
        }
    }
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, language: str = "python"):
        super().__init__(chunk_size, chunk_overlap)
        self.language = language.lower()
        
    def detect_language(self, file_path: str) -> str:
        """
        ファイル拡張子からプログラミング言語を検出
        
        Args:
            file_path: ソースコードファイルパス
            
        Returns:
            検出された言語
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "javascript",
            ".jsx": "javascript",
            ".tsx": "javascript"
        }
        
        return language_map.get(ext, "python")  # デフォルトはPython
    
    def split_text(self, text: str) -> List[str]:
        """
        コードを構造を考慮して分割
        
        Args:
            text: ソースコード
            
        Returns:
            分割されたチャンクのリスト
        """
        if not text:
            return []
            
        # 言語のパターンを取得
        patterns = self.LANGUAGE_PATTERNS.get(self.language, self.LANGUAGE_PATTERNS["python"])
        
        # クラスと関数のパターンを結合
        split_pattern = f"{patterns['class_pattern']}|{patterns['function_pattern']}"
        
        # コードブロックを分割
        blocks = re.split(f'({split_pattern})', text, flags=re.MULTILINE)
        
        chunks = []
        current_block = ""
        
        for i, block in enumerate(blocks):
            # クラスまたは関数の定義行の場合
            if i > 0 and re.match(split_pattern, block, re.MULTILINE):
                # 前のブロックが大きすぎる場合、さらに分割
                if len(current_block) > self.chunk_size:
                    sub_chunks = super().split_text(current_block)
                    chunks.extend(sub_chunks)
                elif current_block:
                    chunks.append(current_block)
                
                # 新しいブロックを開始
                current_block = block
            else:
                # コード部分を現在のブロックに追加
                current_block += block
        
        # 最後のブロックを処理
        if len(current_block) > self.chunk_size:
            sub_chunks = super().split_text(current_block)
            chunks.extend(sub_chunks)
        elif current_block:
            chunks.append(current_block)
            
        return chunks
    
    def split_with_metadata(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        コードを分割し、コード構造情報をメタデータに追加
        
        Args:
            text: ソースコード
            metadata: 基本メタデータ
            
        Returns:
            コンテンツとメタデータを含む辞書のリスト
        """
        if metadata is None:
            metadata = {}
            
        # 言語情報をメタデータに追加
        base_metadata = metadata.copy()
        language = base_metadata.get("language", self.language)
        base_metadata["language"] = language
        
        # ファイルパスからモジュール名を抽出（Pythonの場合）
        if "source" in base_metadata and language == "python":
            file_path = base_metadata["source"]
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            base_metadata["module_name"] = module_name
        
        # テキストを分割
        chunks = self.split_text(text)
        
        result = []
        for i, chunk in enumerate(chunks):
            # 各チャンクのメタデータを作成
            chunk_metadata = base_metadata.copy()
            
            # クラス名または関数名を抽出
            patterns = self.LANGUAGE_PATTERNS.get(language, self.LANGUAGE_PATTERNS["python"])
            
            class_match = re.search(patterns["class_pattern"], chunk, re.MULTILINE)
            function_match = re.search(patterns["function_pattern"], chunk, re.MULTILINE)
            
            if class_match:
                # Python: "class MyClass:" → "MyClass"
                # JavaScript: "class MyClass {" → "MyClass"
                class_name = re.search(r'class (\w+)', class_match.group(1)).group(1)
                chunk_metadata["symbol_type"] = "class"
                chunk_metadata["symbol_name"] = class_name
            elif function_match:
                # 関数名を抽出（言語によって異なる）
                if language == "python":
                    # "def my_func():" → "my_func"
                    function_name = re.search(r'def (\w+)', function_match.group(1)).group(1)
                else:
                    # JavaScriptは複数のパターンがあるため、より複雑な抽出ロジックが必要
                    function_name_match = re.search(
                        r'(function (\w+)|(\w+) =|\b(\w+)\([^\)]*\))', 
                        function_match.group(0)
                    )
                    if function_name_match:
                        for group in function_name_match.groups():
                            if group and not group.startswith('function '):
                                function_name = group.strip()
                                break
                    else:
                        function_name = "anonymous_function"
                
                chunk_metadata["symbol_type"] = "function"
                chunk_metadata["symbol_name"] = function_name
            
            # チャンクインデックス情報
            chunk_metadata.update({
                "chunk_index": i,
                "chunk_total": len(chunks)
            })
            
            result.append({
                "content": chunk,
                "metadata": chunk_metadata
            })
            
        return result