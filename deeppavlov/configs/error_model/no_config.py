from deeppavlov.core.commands.utils import set_deeppavlov_root, expand_path
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.dataset_readers.typos import TyposWikipedia
from deeppavlov.datasets.typos_dataset import TyposDataset
from deeppavlov.models.tokenizers.nltk_tokenizer import NLTKTokenizer
from deeppavlov.vocabs.typos import Wiki100KDictionary
from deeppavlov.models.spellers.error_model.error_model import ErrorModel

from deeppavlov.metrics.accuracy import accuracy

set_deeppavlov_root({})

ds = TyposDataset(TyposWikipedia.read(expand_path('')), test_ratio=0.1)


def lower(batch):
    return [item.lower() for item in batch]


chainer = Chainer(['x'], ['x'], ['y'])

chainer.append(['x'], ['x'], lower)
chainer.append(['y'], ['y'], lower)

tokenizer = NLTKTokenizer()

chainer.append(['x'], ['x'], tokenizer)

model = ErrorModel(Wiki100KDictionary(), save_path='/tmp/error_model')

model.fit(*chainer(*ds.iter_all('train'), to_return=['x', 'y']))
model.save()

chainer.append(['x'], ['x'], model)

x, y = ds.iter_all('test')

y_predicted = chainer(x)

print(accuracy(y, y_predicted))
