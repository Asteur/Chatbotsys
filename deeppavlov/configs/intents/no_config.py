from pathlib import Path
from typing import Iterable

from deeppavlov.core.commands.train import _train_batches
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.data.dataset_iterator import BasicDatasetIterator
from deeppavlov.metrics.accuracy import sets_accuracy
from deeppavlov.models.classifiers.intents.intent_model import KerasIntentModel
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder
from deeppavlov.models.tokenizers.nltk_tokenizer import NLTKTokenizer


class IntentsWrapper:
    def __init__(self, emb_path: [Path, str], classes: Iterable=None,
                 save_path='intents/simple_classifier'):
        self.ready = False

        self.ft = FasttextEmbedder(load_path=emb_path)

        self.classes = classes
        self.save_path = expand_path(save_path)

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

        # self.chainer = Chainer(in_x='text', in_y='classes', out_params='classes_predicted')

    def __call__(self, batch, predict_proba=False):
        if not self.ready:
            raise RuntimeError('Tried to run not fitted classifier. Fit or load it first')
        return self.model(batch, predict_proba=predict_proba)

    def predict(self, batch):
        return self(batch)

    def predict_proba(self, batch):
        return self(batch, predict_proba=True)

    def fit(self, x, y, val_x=None, val_y=None, batch_size=64, epochs=100, patience=5):
        validate = val_x and val_y

        if not self.classes:
            self.classes = {label for labels in y for label in labels}

        data = {
            'train': list(zip(x, y))
        }
        if validate:
            data['valid'] = list(zip(val_x, val_y))

        del x, y, val_x, val_y

        ds_iterator = BasicDatasetIterator(data)

        self.model = KerasIntentModel(self.model_params, self.ft, NLTKTokenizer(), self.classes, mode='train',
                                      save_path=self.save_path)

        train_config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "validation_patience": patience,
            "val_every_n_epochs": validate and 1 or 0,
            "log_every_n_epochs": 1,
            "validate_best": validate,
            "test_best": False
        }

        _train_batches(self.model, ds_iterator, train_config, [('accuracy', sets_accuracy)])

        self.model = KerasIntentModel(self.model_params, self.ft, NLTKTokenizer(), self.classes, mode='infer',
                                      save_path=self.save_path, load_path=self.save_path)
        self.classes = self.model.classes

        self.ready = True

    def load(self, load_path=None):
        if load_path:
            load_path = expand_path(load_path)
        else:
            load_path = self.save_path

        self.model = KerasIntentModel(self.model_params, self.ft, NLTKTokenizer(), self.classes, mode='infer',
                                      save_path=self.save_path, load_path=load_path)
        self.classes = self.model.classes
        self.ready = True
