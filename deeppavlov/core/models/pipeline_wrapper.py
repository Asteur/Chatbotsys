from abc import ABCMeta, abstractmethod

from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


class BasicWrapper(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, generate_batches, batch_size=1, save_dir=None,
            *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, batches_generator):
        pass

    @abstractmethod
    def load(self, load_dir):
        pass

    @abstractmethod
    def save(self, save_dir):
        pass


class NNWrapper(BasicWrapper):
    def __init__(self):
        self.ready = False
        self._epochs_done = 0
        self._batches_seen = 0

    def load(self, load_dir):
        self._prepare_components(load_dir=load_dir)

    def fit(self, generate_batches, batch_size=1, generate_valid_batches=None,
            save_dir=None,
            epochs_limit=-1, batches_limit=-1,
            *args, **kwargs):
        self._prepare_components(generate_batches=generate_batches)
        should_stop = False
        try:
            while True:
                for batch in generate_batches(batch_size=batch_size):
                    self._train_on_batch(batch)
                    self._batches_seen += 1
                    if -1 < batches_limit <= self._batches_seen:
                        should_stop = True
                        break
                if should_stop:
                    break
                self._epochs_done += 1
                if -1 < epochs_limit <= self._epochs_done:
                    break
        except KeyboardInterrupt:
            pass

    @abstractmethod
    def _prepare_components(self, load_dir=None, generate_batches=None, *args, **kwargs):
        pass

    @abstractmethod
    def _train_on_batch(self, batch):
        pass
