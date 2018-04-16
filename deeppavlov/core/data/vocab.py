"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Estimator

log = get_logger(__name__)


@register('default_vocab')
class DefaultVocabulary(Estimator):
    def __init__(self,
                 special_tokens=tuple(),
                 max_tokens=2**30,
                 min_count=1,
                 *args,
                 **kwargs):
        super().__init__(**kwargs)
        self.special_tokens = special_tokens
        self._max_tokens = max_tokens
        self._min_count = min_count
        self.freqs = None
        if self.load_path:
            self.load()

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._i2t[key]
        elif isinstance(key, str):
            return self._t2i[key]
        else:
            raise NotImplementedError("not implemented for type `{}`".format(type(key)))

    def __contains__(self, item):
        return item in self._t2i

    def __len__(self):
        return len(self.freqs)

    def keys(self):
        return (k for k, v in self.freqs.most_common())

    def values(self):
        return (v for k, v in self.freqs.most_common())

    def items(self):
        return self.freqs.most_common()

    def fit(self, tokens):
        self.freqs = Counter(chain(*tokens))
        # toks_freqs = self.freqs.most_common()[:self.]

    def __call__(self, token_batch, **kwargs):
        return [self[s] for s in token_batch]

    def save(self):
        log.info("[saving vocabulary to {}]".format(self.save_path))

        with self.save_path.open('wt') as f:
            for n in range(len(self)):
                token = self._i2t[n]
                cnt = self.freqs[token]
                f.write('{}\t{:d}\n'.format(token, cnt))

    def load(self):
        if self.load_path:
            if self.load_path.is_file():
                log.info("[loading vocabulary from {}]".format(self.load_path))
                tokens, counts = [], []
                for ln in self.load_path.open('r'):
                    token, cnt = ln.split('\t', 1)
                    tokens.append(token)
                    counts.append(int(cnt))
                self._train(tokens=tokens, counts=counts, update=True)
            elif isinstance(self.load_path, Path):
                if not self.load_path.parent.is_dir():
                    raise ConfigError("Provided `load_path` for {} doesn't exist!".format(
                        self.__class__.__name__))
        else:
            raise ConfigError("`load_path` for {} is not provided!".format(self))
