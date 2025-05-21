import numpy as np

class DummyEmbedder:
    def __init__(self, dim=384):
        self.dim = dim

    def embed(self, text):
        # 本来はモデルでベクトル化するが、ここではダミーベクトルを返す
        np.random.seed(hash(text) % (2**32))
        return np.random.rand(self.dim).tolist()

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]