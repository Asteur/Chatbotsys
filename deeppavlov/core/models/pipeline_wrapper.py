import datetime
import itertools
import json
import time
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
    def evaluate(self, batches_iterator):
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

    def fit(self, generate_train_batches, batch_size=1,
            generate_valid_batches=None,
            save_dir=None, load_dir=None,
            epochs_limit=-1, batches_limit=-1,
            log_every_n_batches=-1, log_every_n_epochs=-1, eval_on_n_train_batches=1,
            validate_every_n_epochs=-1,
            *args, **kwargs):
        self._prepare_components(generate_batches=generate_train_batches, batch_size=batch_size, load_dir=load_dir)
        if generate_valid_batches is None and validate_every_n_epochs > 0:
            validate_every_n_epochs = -1
            log.warn('Cannot validate if no generator function is provided for the valid data')
        should_stop = False
        new_epoch = False
        start_time = time.time()
        try:
            while True:
                for batch in generate_train_batches(batch_size=batch_size, shuffle=True):
                    if new_epoch and 0 < log_every_n_epochs and self._epochs_done % log_every_n_epochs == 0\
                            or 0 < log_every_n_batches and self._batches_seen > 0\
                            and self._batches_seen % log_every_n_batches == 0:
                        report = {
                            'epochs_done': self._epochs_done,
                            'batches_seen': self._batches_seen,
                            'time_spent': str(datetime.timedelta(seconds=round(time.time() - start_time + 0.5)))
                        }
                        if eval_on_n_train_batches != 0:
                            if eval_on_n_train_batches == 1:
                                it = iter([batch])
                            else:
                                it = generate_train_batches(batch_size=batch_size, shuffle=True)
                                if eval_on_n_train_batches > 0:
                                    it = itertools.islice(it, eval_on_n_train_batches)
                            report['metrics'] = self.evaluate(it)
                        print(json.dumps(report, ensure_ascii=False))
                    new_epoch = False
                    self._train_on_batch(batch)
                    self._batches_seen += 1
                    if -1 < batches_limit <= self._batches_seen:
                        should_stop = True
                        break
                if should_stop:
                    break
                self._epochs_done += 1
                new_epoch = True
                if -1 < epochs_limit <= self._epochs_done:
                    break
        except KeyboardInterrupt:
            log.info('Interrupted training')

    @abstractmethod
    def _prepare_components(self, load_dir=None, generate_batches=None, batch_size=1, *args, **kwargs):
        pass

    @abstractmethod
    def _train_on_batch(self, batch):
        pass
