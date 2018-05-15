from sklearn.externals import joblib

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Estimator

logger = get_logger(__name__)


@register('skill_tumbler')
class LinearSVCScorer(Estimator):
    def __init__(self, save_path, load_path, vec_path, *args, **kwargs):
        self.save_path = expand_path(save_path)
        self.load_path = expand_path(load_path)
        self.vec_path = expand_path(vec_path)
        self.model = None
        self.vec = None

        if self.load_path.exists():
            self.load()
        else:
            raise RuntimeError('TumblerClassifier: there is no pretrained model')

    def fit(self, *args, **kwargs):
        pass

    def __call__(self, df):
        data = self.vec.transform(df)
        scores = self.model.predict(data)
        return scores

    def load(self, *args, **kwargs):
        logger.info('TumblerClassifier: loading saved model vocab from {}'.format(self.load_path))
        self.model = joblib.load(str(self.load_path))
        self.vec = joblib.load(str(self.vec_path))

    def save(self, *args, **kwargs):
        pass
