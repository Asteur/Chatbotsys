#%% imports
from deeppavlov.core.commands.utils import set_deeppavlov_root, expand_path
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.dataset_readers.typos_reader import TyposWikipedia
from deeppavlov.dataset_iterators.typos_iterator import TyposDatasetIterator
from deeppavlov.models.tokenizers.nltk_tokenizer import NLTKTokenizer
from deeppavlov.vocabs.typos import Wiki100KDictionary
from deeppavlov.models.spellers.error_model.error_model import ErrorModel

from deeppavlov.metrics.accuracy import accuracy
#%% init
set_deeppavlov_root({})

ds = TyposDatasetIterator(TyposWikipedia.read(expand_path('')), test_ratio=0.1)


def lower(batch):
    return [item.lower() for item in batch]


#%% init chainer
chainer = Chainer()

chainer.append(lower)
chainer.append(lower, 'y')

tokenizer = NLTKTokenizer()

chainer.append(tokenizer)
chainer.append(tokenizer, 'y')

model = ErrorModel(Wiki100KDictionary(), save_path='/tmp/error_model')

#%% fit model
model.fit(*zip(*chainer(*ds.get_instances('train'), to_return=['x', 'y'])))
model.save()

#%% append model to chainer
chainer.append(model)

#%% test on test
x, y = ds.get_instances('test')

y_predicted = chainer(x)

print(accuracy(y, y_predicted))

#%% test on input
to_test = 'Helllo'

print(f'{to_test} — {chainer([to_test])[0]}')
