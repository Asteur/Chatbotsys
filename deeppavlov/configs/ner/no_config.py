from deeppavlov.core.commands.utils import set_deeppavlov_root
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.data.vocab import DefaultVocabulary
from deeppavlov.dataset_readers.dstc2_reader import DSTC2DatasetReader
from deeppavlov.dataset_iterators.dstc2_ner_iterator import Dstc2NerDatasetIterator
from deeppavlov.models.ner.ner import NER
from deeppavlov.metrics.accuracy import per_item_accuracy
from deeppavlov.metrics.fmeasure import ner_f1

from deeppavlov.core.commands.train import _train_batches

set_deeppavlov_root({})

ds = Dstc2NerDatasetIterator(DSTC2DatasetReader().read(data_path='dstc2'), dataset_path='dstc2')

chainer = Chainer(in_y='y')

x, y = ds.iter_all('train')

word_vocab = DefaultVocabulary(mode='train',
                               save_path='ner_dstc2_model/word.dict',
                               load_path='ner_dstc2_model/word.dict',
                               level='token')
char_vocab = DefaultVocabulary(mode='train',
                               save_path='ner_dstc2_model/word.dict',
                               load_path='ner_dstc2_model/word.dict',
                               level='char')
tag_vocab = DefaultVocabulary(mode='train',
                              save_path='ner_dstc2_model/word.dict',
                              load_path='ner_dstc2_model/word.dict',
                              level='token')

word_vocab.fit(x)
char_vocab.fit(x)
tag_vocab.fit(y)

ner_params = {"save_path": 'ner_dstc2_model/ner_model',
              "load_path": 'ner_dstc2_model/ner_model',
              "word_vocab": word_vocab, "char_vocab": char_vocab, "tag_vocab": tag_vocab,
              "filter_width": 7,
              "embeddings_dropout": True,
              "n_filters": [128, 128],
              "token_embeddings_dim": 64,
              "char_embeddings_dim": 32,
              "use_batch_norm": True,
              "use_crf": True,
              "learning_rate": 1e-3,
              "dropout_rate": 0.5}

model = NER(**ner_params)

chainer.append(model, in_y='y', main=True)

train_config = {
        'batch_size': 64,

        'metric_optimization': 'maximize',

        'validation_patience': 5,
        'val_every_n_epochs': 3,

        'log_every_n_batches': 0,
        'log_every_n_epochs': 1,
        # 'show_examples': False,

        'validate_best': True,
        'test_best': True
    }

_train_batches(chainer, ds, train_config, [('f1', ner_f1), ('accuracy', per_item_accuracy)])
chainer.load()

x, y = ds.iter_all('valid')
print('valid', ner_f1(y, chainer(x)))

x, y = ds.iter_all('test')
print('test', ner_f1(y, chainer(x)))
