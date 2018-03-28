from deeppavlov.core.common.chainer import Chainer


class Wrapper:
    def __init__(self, *args):
        self.chainer = Chainer()
        self.ready = False

    def __call__(self, *args, **kwargs):
        pass

    def fit(self, generate_batches, batch_size=1):
        self._build_chainer(generate_batches=lambda: generate_batches(batch_size))

    def _batch_pred_eval(self, batch):
        pass

    def evaluate(self, batches_generator):
        pass

    def load(self, path):
        self._build_chainer(pretrained_path=path)

    def _build_chainer(self, pretrained_path=None, generate_batches=None):
        if pretrained_path is None and generate_batches is None:
            raise RuntimeError('Have to have load path or data provider to build the pipeline')
        self.ready = True
