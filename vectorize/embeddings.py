import os
import torch
import logging
import numpy as np
from typing import List, Optional, Union
from huggingface_hub import snapshot_download
import config
from transformers import AutoModel, AutoTokenizer

# ロギング設定
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class DummyEmbedder:
    """テスト用のダミー埋め込みクラス"""
    def __init__(self, dim=384):
        self.dim = dim

    def embed(self, text):
        # 本来はモデルでベクトル化するが、ここではダミーベクトルを返す
        np.random.seed(hash(text) % (2**32))
        return np.random.rand(self.dim).tolist()

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]
        
    def get_dimension(self):
        """埋め込みベクトルの次元数を返す"""
        return self.dim
        
    def embed_query(self, text):
        """
        クエリテキストを埋め込みベクトルに変換するメソッド
        integration_testのno-Docker対応用に追加
        
        Args:
            text: 埋め込むテキスト
            
        Returns:
            埋め込みベクトル（リスト形式）
        """
        # embedメソッドを再利用
        return self.embed(text)

class PLaMoEmbedder:
    """PLaMo-Embedding-1Bを使用したテキスト埋め込みクラス"""
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        PLaMo-Embedding-1Bモデルを初期化します。
        
        Args:
            model_path: モデルのパス。指定がない場合はconfig.EMBEDDING_MODEL_PATHを使用
            device: 推論に使用するデバイス。指定がない場合は自動検出
        """
        self.model_path = model_path or config.EMBEDDING_MODEL_PATH
        
        # Fix Docker path reference if running locally
        if self.model_path.startswith('/app/'):
            # We're running locally, not in Docker
            relative_path = self.model_path[5:]  # Remove /app/ prefix
            self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), relative_path)
            logger.info(f"Detected Docker path, converted to local path: {self.model_path}")
        
        # デバイスを設定（CUDAが使用可能ならGPUを使用）
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # モデルパスが存在しない場合はダウンロード
        if not os.path.exists(self.model_path) or not os.listdir(self.model_path):
            self._download_model()
        
        # モデルとトークナイザの初期化
        try:
            logger.info(f"Loading model from {self.model_path}")
            # Always use self.model_path for both tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=True).to(self.device).eval()
            self.dim = self.model.config.hidden_size
            logger.info(f"Model loaded successfully. Embedding dimension: {self.dim}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _download_model(self):
        """HuggingFaceからモデルをダウンロード"""
        try:
            # Try to use the dedicated script for downloading the model
            script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "ensure_model.py")
            if os.path.exists(script_path):
                logger.info("Using dedicated model download script")
                import sys
                sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                from scripts.ensure_model import ensure_model_downloaded
                
                success = ensure_model_downloaded(self.model_path)
                if not success:
                    raise RuntimeError("Failed to download model using ensure_model script")
                return
            
            # Fallback to direct download if script doesn't exist
            logger.info(f"Downloading PLaMo-Embedding-1B model to {self.model_path}")
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Use explicit parameters for more reliable download
            snapshot_download(
                repo_id="pfnet/plamo-embedding-1b", 
                local_dir=self.model_path,
                local_dir_use_symlinks=False,  # Use actual files, not symlinks
                revision="main",  # Use the main branch
                ignore_patterns=["*.h5", "*.ot", "*.msgpack"],  # Skip unnecessary large files
                max_workers=2  # Use fewer workers to avoid connection issues
            )
            
            # Verify if essential files were downloaded
            files = os.listdir(self.model_path)
            logger.info(f"Model files downloaded: {files}")
            
            # Check if model weights file exists
            if not any(f in files for f in ["pytorch_model.bin", "model.safetensors"]):
                raise RuntimeError("Model weight files not found after download")
                
            logger.info("Model downloaded successfully")
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        平均プーリングによるトークン表現からの文書埋め込みの取得
        
        Args:
            model_output: モデルの出力
            attention_mask: アテンションマスク
            
        Returns:
            文書埋め込みベクトル
        """
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        
        # attention_maskをfloatに変換して拡張
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        
        # トークン埋め込みとマスクの要素ごとの積を計算
        masked_embeddings = token_embeddings * input_mask_expanded
        
        # 各シーケンスの合計を計算
        summed = torch.sum(masked_embeddings, dim=1)
        
        # 非パディングトークンの数を計算
        counts = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # 平均計算（合計 ÷ カウント）
        mean_pooled = summed / counts
        
        return mean_pooled
    
    @torch.no_grad()
    def embed(self, text: str) -> List[float]:
        """
        単一テキストの埋め込みベクトルを計算
        
        Args:
            text: 埋め込むテキスト
            
        Returns:
            埋め込みベクトル（リスト形式）
        """
        # 入力のエンコード
        encoded_input = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        # モデル推論
        outputs = self.model(**encoded_input)
        
        # 平均プーリングで埋め込みを取得
        embeddings = self._mean_pooling(outputs, encoded_input["attention_mask"])
        
        # 正規化
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # NumPyに変換してからリストに変換
        return embeddings[0].cpu().numpy().tolist()
    
    def get_dimension(self) -> int:
        """埋め込みベクトルの次元数を返す"""
        return self.dim
        
    @torch.no_grad()
    def embed_batch(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        """
        複数テキストのバッチ処理による埋め込みベクトルの計算
        
        Args:
            texts: 埋め込むテキストのリスト
            batch_size: バッチサイズ
            
        Returns:
            埋め込みベクトルのリスト
        """
        # 入力テキストの数が多い場合は、小さいバッチサイズを使用
        actual_batch_size = min(batch_size, 8) if len(texts) > 100 else batch_size
        if len(texts) > 100 and actual_batch_size < batch_size:
            logger.info(f"Large input detected ({len(texts)} texts), reducing batch size to {actual_batch_size}")
            
        all_embeddings = []
        
        # バッチごとに処理
        for i in range(0, len(texts), actual_batch_size):
            batch_texts = texts[i:i+actual_batch_size]
            
            # 進捗を表示
            if i % (actual_batch_size * 10) == 0 or i + actual_batch_size >= len(texts):
                logger.info(f"Embedding batch {i//actual_batch_size + 1}/{(len(texts)-1)//actual_batch_size + 1} ({i}/{len(texts)} texts)")
            
            try:
                # バッチをエンコード
                encoded_inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors="pt"
                ).to(self.device)
                
                # モデル推論
                outputs = self.model(**encoded_inputs)
                
                # 平均プーリングで埋め込みを取得
                embeddings = self._mean_pooling(outputs, encoded_inputs["attention_mask"])
                
                # 正規化
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                # NumPyに変換してからリストに変換し、結果に追加
                batch_embeddings = embeddings.cpu().numpy().tolist()
                all_embeddings.extend(batch_embeddings)
                
                # 明示的にメモリを解放
                del encoded_inputs
                del outputs
                del embeddings
                del batch_embeddings
                
                # 大きなバッチの場合はメモリ使用量を減らすためにGCを呼び出す
                if len(texts) > 1000 and i % (actual_batch_size * 50) == 0:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error embedding batch starting at index {i}: {e}")
                # エラーの場合はダミーの埋め込みを提供
                dummy_embeddings = [[0.0] * self.dim] * len(batch_texts)
                all_embeddings.extend(dummy_embeddings)
        
        return all_embeddings