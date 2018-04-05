import json
from pathlib import Path
from typing import Iterable

from deeppavlov.core.commands.train import _train_batches
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.data.dataset_iterator import BasicDatasetIterator
from deeppavlov.core.models.pipeline_wrapper import NNWrapper
from deeppavlov.metrics.accuracy import sets_accuracy
from deeppavlov.models.classifiers.intents.intent_model import KerasIntentModel
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder
from deeppavlov.models.tokenizers.nltk_tokenizer import NLTKTokenizer


class IntentsNNWrapper(NNWrapper):
    def __init__(self, emb_path: [Path, str], classes: Iterable = None,
                 save_dir='intents/simple_classifier'):
        super().__init__(save_dir)

        self.ft = FasttextEmbedder(load_path=emb_path)

        self.classes = classes
        self.save_dir = expand_path(save_dir)

        self.model: KerasIntentModel = None
        self.model_params = {
            "kernel_sizes_cnn": [
                1,
                2,
                3
            ],
            "filters_cnn": 256,
            "lear_metrics": [
                "binary_accuracy",
                "fmeasure"
            ],
            "confident_threshold": 0.5,
            "optimizer": "Adam",
            "lear_rate": 0.01,
            "lear_rate_decay": 0.1,
            "loss": "binary_crossentropy",
            "text_size": 15,
            "coef_reg_cnn": 1e-4,
            "coef_reg_den": 1e-4,
            "dropout_rate": 0.5,
            "epochs": 1000,
            "dense_size": 100,
            "model_name": "cnn_model"
        }

    def _train_on_batch(self, batch):
        self.model.train_on_batch(*batch)

    def evaluate(self, batches_iterator):
        examples_len = 0
        correct = 0
        for x, y_true in batches_iterator:
            examples_len += len(y_true)
            y_predicted = self(x)
            correct += sum([set(y1) == set(y2) for y1, y2 in zip(y_true, y_predicted)])

        accuracy = correct / examples_len if examples_len > 0 else 0
        return {'accuracy': accuracy}

    def __call__(self, batch, predict_proba=False):
        if not self.ready:
            raise RuntimeError('Tried to run not fitted classifier. Fit or load it first')
        return self.model(batch, predict_proba=predict_proba)

    def predict(self, batch):
        return self(batch)

    def predict_proba(self, batch):
        return self(batch, predict_proba=True)

    def save(self, save_dir=None):
        if save_dir:
            save_dir = expand_path(save_dir)
        with self.save_dir.joinpath('classes.json').open('w') as f:
            json.dump(self.classes, f)
        self.model.save(save_dir / 'model')

    def _prepare_components(self, load_dir=None, generate_batches=None, batch_size=1, *args, **kwargs):
        self.save_dir = expand_path(self.save_dir)
        model_load_path = None
        if load_dir is not None:
            load_dir = expand_path(load_dir)
            model_load_path = load_dir / 'model'
            with load_dir.joinpath('classes.json').open as f:
                self.classes = json.load(f)

        if generate_batches:
            mode = 'train'
            if self.classes is None:
                self.classes = {label
                                for _, y in generate_batches(batch_size=1, shuffle=False)
                                for labels in y for label in labels}
        else:
            mode = 'infer'

        self.model = KerasIntentModel(self.model_params, self.ft, NLTKTokenizer(), self.classes, mode=mode,
                                      save_path=self.save_dir / 'model', load_path=model_load_path)
        self.classes = self.model.classes
