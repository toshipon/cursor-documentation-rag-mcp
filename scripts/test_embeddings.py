import sys
import os
import time
import logging

# カレントディレクトリをプロジェクトルートに設定するためのパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorize.embeddings import PLaMoEmbedder, DummyEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dummy_embedder():
    """ダミー埋め込みのテスト"""
    logger.info("テスト: DummyEmbedder")
    embedder = DummyEmbedder(dim=384)
    
    test_text = "これはテストテキストです。"
    
    # 単一テキストのベクトル化
    start_time = time.time()
    vector = embedder.embed(test_text)
    elapsed = time.time() - start_time
    
    logger.info(f"単一テキストのベクトル化: {elapsed:.4f}秒")
    logger.info(f"ベクトルの次元数: {len(vector)}")
    logger.info(f"ベクトルの一部: {vector[:5]}...")
    
    # バッチベクトル化
    test_texts = [
        "これはテスト文章1です。",
        "This is test sentence 2.",
        "これは少し長めのテスト文章です。複数の文を含んでいます。ベクトル化の性能を確認します。"
    ]
    
    start_time = time.time()
    vectors = embedder.embed_batch(test_texts)
    elapsed = time.time() - start_time
    
    logger.info(f"バッチテキストのベクトル化 ({len(test_texts)}テキスト): {elapsed:.4f}秒")
    logger.info(f"ベクトル数: {len(vectors)}")
    logger.info(f"各ベクトルの次元数: {[len(v) for v in vectors]}")

def test_plamo_embedder():
    """PLaMo-Embedding-1Bの埋め込みテスト"""
    logger.info("テスト: PLaMoEmbedder")
    
    try:
        # モデルの初期化（必要に応じてダウンロード）
        start_time = time.time()
        embedder = PLaMoEmbedder()
        init_time = time.time() - start_time
        logger.info(f"モデル初期化時間: {init_time:.2f}秒")
        
        test_text = "これはPLaMo-Embedding-1Bのテストです。日本語の埋め込み性能を確認します。"
        
        # 単一テキストのベクトル化
        start_time = time.time()
        vector = embedder.embed(test_text)
        elapsed = time.time() - start_time
        
        logger.info(f"単一テキストのベクトル化: {elapsed:.4f}秒")
        logger.info(f"ベクトルの次元数: {len(vector)}")
        logger.info(f"ベクトルの一部: {vector[:5]}...")
        
        # バッチベクトル化
        test_texts = [
            "これはテスト文章1です。",
            "This is test sentence 2.",
            "これは少し長めのテスト文章です。複数の文を含んでいます。ベクトル化の性能を確認します。",
            "コードのドキュメントをベクトル化することで、関連コードの検索が容易になります。",
            "PLaMo-Embedding-1Bは日本語と英語の両方に対応しています。"
        ]
        
        start_time = time.time()
        vectors = embedder.embed_batch(test_texts)
        elapsed = time.time() - start_time
        
        logger.info(f"バッチテキストのベクトル化 ({len(test_texts)}テキスト): {elapsed:.4f}秒")
        logger.info(f"ベクトル数: {len(vectors)}")
        logger.info(f"各ベクトルの次元数: {[len(v) for v in vectors]}")
        
        # 類似度テスト
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # ベクトルをNumPy配列に変換
        np_vectors = np.array(vectors)
        
        # 各ベクトル間のコサイン類似度を計算
        similarities = cosine_similarity(np_vectors)
        
        logger.info("テキスト間の類似度マトリックス:")
        for i in range(len(test_texts)):
            for j in range(i+1, len(test_texts)):
                logger.info(f"類似度 ({i+1}-{j+1}): {similarities[i, j]:.4f}")
                logger.info(f"  テキスト{i+1}: {test_texts[i][:50]}...")
                logger.info(f"  テキスト{j+1}: {test_texts[j][:50]}...")
                
    except Exception as e:
        logger.error(f"PLaMoEmbedderテスト中にエラーが発生: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # テスト実行
    try:
        # まずダミー埋め込みをテスト
        test_dummy_embedder()
        
        print("\n" + "="*80 + "\n")
        
        # 次にPLaMo埋め込みをテスト
        test_plamo_embedder()
    except Exception as e:
        logger.error(f"テスト実行中にエラーが発生: {e}")
        import traceback
        logger.error(traceback.format_exc())
