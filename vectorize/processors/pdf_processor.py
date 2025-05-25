import os
import logging
import pdfplumber
from typing import List, Dict, Any, Tuple, Optional
import re
from collections import defaultdict
import unicodedata
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class TableMetrics:
    """テーブルの品質メトリクス"""
    column_count: int
    row_count: int
    data_density: float  # 空でないセルの割合
    alignment_score: float  # 列の位置揃えスコア
    structure_confidence: float  # 構造的な信頼度
    content_quality: float  # コンテンツの品質スコア
    japanese_ratio: float = 0.0  # 日本語文字の割合
    layout_score: float = 0.0  # レイアウトの品質スコア

@dataclass
class TableStructure:
    """テーブルの構造情報"""
    columns: int
    header_row: int
    rows: int
    alignments: List[str]
    content_types: List[str]
    metrics: TableMetrics
    title: str = ""
    features: Dict[str, Any] = field(default_factory=dict)

class PDFTableAnalyzer:
    """PDFのテーブル解析を行うクラス"""
    
    # 文字種の定義
    CHAR_TYPES = {
        'kanji': (0x4E00, 0x9FFF),      # CJK統合漢字
        'hiragana': (0x3040, 0x309F),    # ひらがな
        'katakana': (0x30A0, 0x30FF),    # カタカナ
        'full_width': (0xFF00, 0xFFEF),  # 全角英数字
    }
    
    # テーブル検出パターン
    TABLE_PATTERNS = {
        'headers': [
            # 日本語のヘッダーパターン
            r'^(番号|No\.|ID|＃)\s*[:：]?\s*',
            r'^(項目|内容|説明|概要|名称|タイトル)\s*[:：]?\s*',
            r'^(種類|タイプ|型|カテゴリ)\s*[:：]?\s*',
            r'^(値|データ|結果|状態)\s*[:：]?\s*',
            r'^(日時|日付|期間)\s*[:：]?\s*',
            # 英語のヘッダーパターン
            r'^(Number|No\.|ID|#)\s*:?\s*',
            r'^(Item|Name|Title|Description)\s*:?\s*',
            r'^(Type|Category|Class)\s*:?\s*',
            r'^(Value|Data|Result|Status)\s*:?\s*',
            r'^(Date|Time|Period)\s*:?\s*'
        ],
        'markers': [
            # 日本語のテーブルマーカー
            '表', 'テーブル', '一覧', '比較', '対照',
            '項目', 'リスト', '仕様', '設定', 'パラメータ',
            'オプション', '属性', '定義', '概要', '分類',
            # 英語のテーブルマーカー
            'Table', 'List', 'Comparison', 'Parameters',
            'Options', 'Properties', 'Attributes', 'Settings'
        ],
        'separators': ['|', '┃', '｜', '│', '┆', '┊', '：', ':'],
        'borders': ['-', '=', '─', '━', '—', '_', '＿', '―']
    }
    
    def _get_char_type(self, char: str) -> str:
        """文字の種類を判定"""
        code = ord(char)
        for type_name, (start, end) in self.CHAR_TYPES.items():
            if start <= code <= end:
                return type_name
        return 'other'

    def _analyze_text_properties(self, text: str) -> Dict[str, float]:
        """テキストの特徴を分析"""
        if not text:
            return {
                'japanese_ratio': 0.0,
                'space_ratio': 0.0,
                'symbol_ratio': 0.0,
                'alpha_ratio': 0.0,
                'digit_ratio': 0.0
            }
            
        char_counts = defaultdict(int)
        total_chars = len(text)
        
        for char in text:
            if char.isspace():
                char_counts['space'] += 1
            elif char.isalpha():
                char_counts['alpha'] += 1
            elif char.isdigit():
                char_counts['digit'] += 1
            elif unicodedata.category(char).startswith('P'):
                char_counts['symbol'] += 1
            else:
                char_type = self._get_char_type(char)
                char_counts[char_type] += 1
        
        japanese_chars = sum(char_counts[t] for t in ['kanji', 'hiragana', 'katakana', 'full_width'])
        
        return {
            'japanese_ratio': japanese_chars / total_chars,
            'space_ratio': char_counts['space'] / total_chars,
            'symbol_ratio': char_counts['symbol'] / total_chars,
            'alpha_ratio': char_counts['alpha'] / total_chars,
            'digit_ratio': char_counts['digit'] / total_chars
        }

    def _detect_table_boundary(self, lines: List[str], start_idx: int) -> Optional[Tuple[int, int]]:
        """テーブルの開始位置と終了位置を検出"""
        if start_idx >= len(lines):
            return None
            
        # テーブルの特徴を評価するスコア関数
        def score_line(line: str) -> float:
            if not line.strip():
                return 0.0
                
            score = 0.0
            # 区切り文字の存在
            for sep in self.TABLE_PATTERNS['separators']:
                if sep in line:
                    score += 0.3
                    break
            
            # 整形された空白
            if re.search(r'\s{2,}', line):
                score += 0.2
            
            # 列のアライメント
            cells = self._split_line(line)
            if len(cells) >= 2:
                score += 0.2
            
            # 数値の存在
            if re.search(r'\d+', line):
                score += 0.1
            
            return score
        
        # テーブルの開始を探索
        scores = [score_line(line) for line in lines[start_idx:]]
        if not scores:
            return None
            
        # スコアの移動平均を計算
        window_size = 3
        avg_scores = []
        for i in range(len(scores) - window_size + 1):
            avg_scores.append(sum(scores[i:i+window_size]) / window_size)
        
        # テーブルの境界を決定
        threshold = 0.3
        table_start = None
        table_end = None
        
        for i, score in enumerate(avg_scores):
            if score > threshold:
                if table_start is None:
                    table_start = start_idx + i
            elif table_start is not None and table_end is None:
                table_end = start_idx + i
                break
        
        if table_start is not None:
            table_end = table_end or (start_idx + len(scores))
            return table_start, table_end
            
        return None

    def _normalize_table_content(self, cells: List[List[str]]) -> List[List[str]]:
        """テーブルの内容を正規化"""
        if not cells:
            return []
            
        # 空のセルを標準化
        normalized = [[cell.strip() if cell else '' for cell in row] for row in cells]
        
        # 列数を統一
        max_cols = max(len(row) for row in normalized)
        normalized = [row + [''] * (max_cols - len(row)) for row in normalized]
        
        # セルの内容をクリーニング
        cleaned = []
        for row in normalized:
            cleaned_row = []
            for cell in row:
                # 改行を空白に置換
                cell = cell.replace('\n', ' ')
                # 連続する空白を1つに
                cell = ' '.join(cell.split())
                # 全角数字を半角に
                cell = unicodedata.normalize('NFKC', cell)
                cleaned_row.append(cell)
            cleaned.append(cleaned_row)
        
        return cleaned

    def process_table(self, lines: List[str]) -> Tuple[Optional[TableStructure], Optional[str]]:
        """テーブルを処理してMarkdown形式に変換"""
        if not lines:
            return None, ""
            
        # テーブル構造の検出
        boundary = self._detect_table_boundary(lines, 0)
        if not boundary:
            return None, ""
            
        start_idx, end_idx = boundary
        table_lines = lines[start_idx:end_idx]
        
        # テーブルのパース
        cells = [self._split_line(line) for line in table_lines]
        cells = [row for row in cells if row]  # 空行を除去
        
        if len(cells) < 2:  # ヘッダー + 最低1行のデータ
            return None, ""
        
        # テーブルの内容を正規化
        normalized = self._normalize_table_content(cells)
        if not normalized:
            return None, ""
        
        # 列の解析
        col_count = len(normalized[0])
        alignments = []
        content_types = []
        
        for col in range(col_count):
            col_data = [row[col] for row in normalized]
            alignments.append(self._detect_alignment(col_data))
            content_types.append(self._detect_content_type(col_data))
        
        # テーブルメトリクスの計算
        metrics = self._calculate_table_metrics(normalized)
        
        # 構造情報の作成
        structure = TableStructure(
            columns=col_count,
            header_row=0,  # デフォルトは最初の行をヘッダーとする
            rows=len(normalized),
            alignments=alignments,
            content_types=content_types,
            metrics=metrics,
            title=self._find_table_title(lines[:start_idx])
        )
        
        # Markdown形式に変換
        markdown = self._format_as_markdown(structure, normalized)
        
        return structure, markdown

    def _format_as_markdown(self, structure: TableStructure, data: List[List[str]]) -> str:
        """テーブルをMarkdown形式に変換"""
        if not data:
            return ""
        
        lines = []
        
        # タイトル
        if structure.title:
            lines.append(f"\n**{structure.title}**\n")
        
        # 列幅の計算（日本語文字を考慮）
        col_widths = []
        for col in range(structure.columns):
            width = max(
                sum(2 if ord(c) > 127 else 1 for c in str(row[col]))
                for row in data
            )
            col_widths.append(max(3, min(width, 30)))  # 最小3文字、最大30文字
        
        # ヘッダー行
        header = "| "
        header += " | ".join(
            str(cell).ljust(width) for cell, width in zip(data[0], col_widths)
        )
        header += " |"
        lines.append(header)
        
        # 区切り行
        separator = "| "
        for i, (width, align) in enumerate(zip(col_widths, structure.alignments)):
            if align == 'right':
                separator += "-" * (width-1) + ":" + " | "
            else:
                separator += "-" * width + " | "
        lines.append(separator.rstrip())
        
        # データ行
        for row in data[1:]:
            row_str = "| "
            row_str += " | ".join(
                str(cell).ljust(width) for cell, width in zip(row, col_widths)
            )
            row_str += " |"
            lines.append(row_str)
        
        return "\n".join(lines)

    def _detect_alignment(self, values: List[str]) -> str:
        """列の位置揃えを検出"""
        if not values:
            return 'left'
            
        # 数値の割合を計算
        numeric_count = sum(
            1 for v in values 
            if re.match(r'^[-+]?\d*\.?\d+$', v) or
               re.match(r'^\d{4}[-/年]\d{1,2}[-/月]\d{1,2}', v)
        )
        
        return 'right' if numeric_count / len(values) > 0.5 else 'left'

    def _detect_content_type(self, values: List[str]) -> str:
        """列のコンテンツタイプを検出"""
        if not values:
            return 'text'
            
        type_scores = defaultdict(int)
        
        for value in values:
            if not value.strip():
                continue
                
            # 数値
            if re.match(r'^[-+]?\d*\.?\d+$', value):
                type_scores['numeric'] += 1
            # 日付
            elif re.match(r'^\d{4}[-/年]\d{1,2}[-/月]\d{1,2}', value):
                type_scores['date'] += 1
            # 日本語
            elif any(ord(c) > 0x3040 for c in value):
                type_scores['japanese'] += 1
            # コード
            elif re.match(r'^[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)*$', value):
                type_scores['code'] += 1
            else:
                type_scores['text'] += 1
        
        if not type_scores:
            return 'text'
            
        return max(type_scores.items(), key=lambda x: x[1])[0]

    def _calculate_table_metrics(self, data: List[List[str]]) -> TableMetrics:
        """テーブルの品質メトリクスを計算"""
        if not data:
            return TableMetrics(
                column_count=0,
                row_count=0,
                data_density=0.0,
                alignment_score=0.0,
                structure_confidence=0.0,
                content_quality=0.0
            )
        
        # データの密度
        total_cells = len(data) * len(data[0])
        non_empty_cells = sum(1 for row in data for cell in row if cell.strip())
        data_density = non_empty_cells / total_cells
        
        # 日本語の割合
        jp_chars = sum(
            1 for row in data for cell in row 
            for c in cell if 0x3040 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FFF
        )
        total_chars = sum(len(cell) for row in data for cell in row)
        japanese_ratio = jp_chars / total_chars if total_chars > 0 else 0.0
        
        # レイアウトスコア
        layout_scores = []
        for i in range(len(data[0])):
            col_data = [row[i] for row in data]
            # 列の整列性
            alignment = self._detect_alignment(col_data)
            aligned_count = sum(1 for v in col_data if self._check_alignment(v, alignment))
            layout_scores.append(aligned_count / len(col_data))
        
        layout_score = sum(layout_scores) / len(layout_scores) if layout_scores else 0.0
        
        # 構造的な信頼度
        structure_confidence = self._calculate_structure_confidence(data)
        
        # コンテンツ品質
        content_quality = self._calculate_content_quality(data)
        
        return TableMetrics(
            column_count=len(data[0]),
            row_count=len(data),
            data_density=data_density,
            alignment_score=layout_score,
            structure_confidence=structure_confidence,
            content_quality=content_quality,
            japanese_ratio=japanese_ratio,
            layout_score=layout_score
        )

    def _calculate_structure_confidence(self, data: List[List[str]]) -> float:
        """テーブル構造の信頼度を計算"""
        if not data:
            return 0.0
        
        scores = []
        
        # ヘッダーの品質
        header = data[0]
        header_score = sum(1 for cell in header if cell.strip()) / len(header)
        scores.append(header_score)
        
        # データ行の一貫性
        for row in data[1:]:
            # 空でないセルの割合
            non_empty = sum(1 for cell in row if cell.strip())
            row_score = non_empty / len(row)
            scores.append(row_score)
        
        # 列の一貫性
        for col in range(len(data[0])):
            col_data = [row[col] for row in data]
            col_type = self._detect_content_type(col_data)
            type_matches = sum(
                1 for v in col_data 
                if self._check_content_type(v, col_type)
            )
            col_score = type_matches / len(col_data)
            scores.append(col_score)
        
        return sum(scores) / len(scores) if scores else 0.0

    def _check_content_type(self, value: str, expected_type: str) -> bool:
        """値が期待されるコンテンツタイプに一致するか確認"""
        if not value.strip():
            return True
            
        if expected_type == 'numeric':
            return bool(re.match(r'^[-+]?\d*\.?\d+$', value))
        elif expected_type == 'date':
            return bool(re.match(r'^\d{4}[-/年]\d{1,2}[-/月]\d{1,2}', value))
        elif expected_type == 'japanese':
            return any(0x3040 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FFF for c in value)
        elif expected_type == 'code':
            return bool(re.match(r'^[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)*$', value))
            
        return True

    def _calculate_content_quality(self, data: List[List[str]]) -> float:
        """テーブルコンテンツの品質スコアを計算"""
        if not data:
            return 0.0
        
        scores = []
        
        for row in data:
            for cell in row:
                if not cell.strip():
                    continue
                
                # セルの品質を評価
                cell_score = 1.0
                
                # 長すぎる値にペナルティ
                if len(cell) > 100:
                    cell_score *= 0.7
                # 短すぎる値にペナルティ
                elif len(cell) < 2:
                    cell_score *= 0.8
                
                # 異常な文字の存在をチェック
                if re.search(r'[^\w\s\-_.,;:()[\]{}！？、。（）［］｛｝]', cell):
                    cell_score *= 0.9
                
                # 日本語文字の適切な使用
                jp_ratio = sum(1 for c in cell if 0x3040 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FFF) / len(cell)
                if 0 < jp_ratio < 1:  # 日本語と英数字が混在
                    cell_score *= 0.95
                
                scores.append(cell_score)
        
        return sum(scores) / len(scores) if scores else 0.0

    def _split_line(self, line: str) -> List[str]:
        """行をセルに分割"""
        line = line.strip()
        if not line:
            return []
        
        # 区切り文字による分割
        for sep in self.TABLE_PATTERNS['separators']:
            if sep in line:
                return [cell.strip() for cell in line.split(sep) if cell.strip()]
        
        # 空白文字による分割
        parts = re.split(r'\s{2,}', line)
        return [part.strip() for part in parts if part.strip()]

    def _check_alignment(self, value: str, expected_align: str) -> bool:
        """値の位置揃えが期待通りかチェック"""
        value = value.strip()
        if not value:
            return True
        
        if expected_align == 'right':
            return bool(re.match(r'^[-+]?\d*\.?\d+$', value) or
                       re.match(r'^\d{4}[-/年]\d{1,2}[-/月]\d{1,2}', value))
        
        return True

    def _find_table_title(self, context_lines: List[str], max_lines: int = 3) -> str:
        """テーブルのタイトルを探す"""
        if not context_lines:
            return ""
        
        # 最後のmax_lines行を逆順に検索
        for line in reversed(context_lines[-max_lines:]):
            line = line.strip()
            if not line:
                continue
            
            # マーカーを含む行を探す
            for marker in self.TABLE_PATTERNS['markers']:
                if marker in line:
                    return line
        
        return ""

class PDFProcessor:
    """
    Processes PDF files to extract text and metadata.
    """

    def __init__(self):
        """
        Initializes the PDFProcessor.
        """
        # Logger setup can be done here if specific configuration is needed
        pass

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text and metadata from each page of a PDF file.

        Args:
            file_path: The path to the PDF file.

        Returns:
            A list of dictionaries, where each dictionary represents a page
            and contains the extracted text and metadata (source and page number).
            Returns an empty list if the file cannot be processed or is empty.
        """
        processed_chunks = []
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return processed_chunks

        try:
            # Open the PDF file using pdfplumber
            with pdfplumber.open(file_path) as pdf:
                if not pdf.pages:
                    logger.warning(f"No pages found in PDF: {file_path}")
                    return processed_chunks
                
                # Iterate through each page in the PDF
                for i, page in enumerate(pdf.pages):
                    page_number = i + 1  # Page numbers are 1-indexed
                    
                    # Extract text from the page
                    text = page.extract_text()
                    
                    if text and text.strip():
                        # Construct the metadata dictionary
                        metadata = {
                            'source': file_path,
                            'page_number': page_number
                        }
                        
                        # Append the processed chunk to the list
                        processed_chunks.append({
                            'text': text.strip(),
                            'metadata': metadata
                        })
                    else:
                        logger.info(f"No text extracted from page {page_number} of {file_path}")

        except pdfplumber.exceptions.PDFSyntaxError as e:
            logger.error(f"PDFSyntaxError processing file {file_path}: {e}")
        except Exception as e:
            # Catch any other exceptions during PDF processing
            logger.error(f"Error processing PDF file {file_path}: {e}")
            # Depending on the desired behavior, you might want to re-raise
            # or return partially processed data if applicable.
            # For now, returning an empty list or whatever was processed so far.

        return processed_chunks

# Example Usage (optional, for testing purposes)
if __name__ == '__main__':
    # Configure basic logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Create a dummy PDF file for testing
    # In a real scenario, you would have a PDF file path.
    dummy_pdf_path = "sample_documents/sample.pdf" # Assuming this was created in a previous step
    
    # Create dummy PDF if it doesn't exist (simplified for example)
    if not os.path.exists(dummy_pdf_path):
        os.makedirs("sample_documents", exist_ok=True)
        try:
            with pdfplumber.open(dummy_pdf_path, "w") as pdf: # This is not how pdfplumber creates PDFs
                 # This is a placeholder. pdfplumber is for reading.
                 # For a real test, you'd need an actual PDF file.
                 # For now, we'll simulate a simple text extraction.
                 pass
            # For testing, let's assume a simple text extraction logic here
            # or use a pre-existing PDF. The following lines would be part of PDF creation.
            # For this example, we will rely on the PDF created in the previous subtask.
            logger.info(f"Created a dummy PDF for testing: {dummy_pdf_path} - Please replace with a real PDF.")

    if os.path.exists(dummy_pdf_path):
        processor = PDFProcessor()
        extracted_data = processor.process_file(dummy_pdf_path)
        
        if extracted_data:
            logger.info(f"Successfully extracted data from {dummy_pdf_path}:")
            for chunk in extracted_data:
                logger.info(f"  Page {chunk['metadata']['page_number']}: {chunk['text'][:100]}...") # Print first 100 chars
        else:
            logger.warning(f"No data extracted from {dummy_pdf_path}. Ensure it's a valid PDF with text.")
    else:
        logger.error(f"Test PDF file not found: {dummy_pdf_path}. Cannot run example.")

# Ensure pdfplumber is installed
# You might need to run: pip install pdfplumber
