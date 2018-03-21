from pathlib import Path

from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder


class IntentsWrapper:
    def __init__(self, emb_path: [Path, str]):
        self.ft = FasttextEmbedder(load_path=emb_path)
        self.ft.load()

    def __call__(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        return self(*args, **kwargs)

    def fit(self, x, y, val_x=None, val_y=None):
        pass
