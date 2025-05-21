import os
import torch
import logging
import numpy as np
from typing import List, Optional, Union
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download
import config

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
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).to(self.device).eval()
            self.dim = self.model.config.hidden_size
            logger.info(f"Model loaded successfully. Embedding dimension: {self.dim}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _download_model(self):
        """HuggingFaceからモデルをダウンロード"""
        try:
            logger.info(f"Downloading PLaMo-Embedding-1B model to {self.model_path}")
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            snapshot_download(repo_id="pfnet/plamo-embedding-1b", local_dir=self.model_path)
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
        all_embeddings = []
        
        # バッチごとに処理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
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
        
        return all_embeddings