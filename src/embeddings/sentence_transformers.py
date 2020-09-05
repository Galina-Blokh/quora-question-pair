import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, models as sent_models
from definitions import DEFAULT_DEVICE


class SentenceBertEmbeddings:
    def __init__(self, bert_path):
        word_embedding_model = sent_models.Transformer(bert_path)
        pooling_model = sent_models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                            pooling_mode_mean_tokens=True,
                                            pooling_mode_cls_token=False,
                                            pooling_mode_max_tokens=False)
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        self.model.to(DEFAULT_DEVICE)
        self.model.eval()

    def text_vector(self, sentences):
        return np.stack(self.model.encode([sentences], show_progress_bar=True))


if __name__ == "__main__":
    import fire
