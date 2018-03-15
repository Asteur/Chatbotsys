from deeppavlov.core.commands.utils import set_deeppavlov_root, expand_path
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.dataset_readers.typos_reader import TyposWikipedia
from deeppavlov.dataset_iterators.typos_iterator import TyposDatasetIterator
from deeppavlov.models.tokenizers.nltk_tokenizer import NLTKTokenizer
from deeppavlov.vocabs.typos import Wiki100KDictionary
from deeppavlov.models.spellers.error_model.error_model import ErrorModel

from deeppavlov.metrics.accuracy import accuracy

set_deeppavlov_root({})

ds = TyposDatasetIterator(TyposWikipedia.read(expand_path('')), test_ratio=0.1)


def lower(batch):
    return [item.lower() for item in batch]


chainer = Chainer()

chainer.append(lower)
chainer.append(lower, 'y')

tokenizer = NLTKTokenizer()

chainer.append(tokenizer)

model = ErrorModel(Wiki100KDictionary(), save_path='/tmp/error_model')

model.fit(*chainer(*ds.iter_all('train'), to_return=['x', 'y']))
model.save()

chainer.append(model)

x, y = ds.iter_all('test')

y_predicted = chainer(x)

print(accuracy(y, y_predicted))

to_test = 'Helllo'

print(f'{to_test} — {chainer([to_test])[0]}')
