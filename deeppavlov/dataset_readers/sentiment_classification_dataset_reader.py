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

import logging
from pathlib import Path

import pandas as pd
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download, mark_done

logger = logging.getLogger(__name__)


@register('sentiment_datasetreader')
class SentimentDatasetReader(DatasetReader):
    """
    Class provides reading dataset in .csv format
    """

    url = 'http://lnsigo.mipt.ru/export/datasets/sentiment/airline_tweets.csv'

    @overrides
    def read(self, data_path):
        """
        Read dataset from data_path directory.
        Args:
            data_path: directory with files

        Returns:
            dictionary with types from train, test, & valid.
            Each field of dictionary is a list of tuples (x_i, y_i)
        """
        filename = "dataset"
        if not Path(data_path).joinpath(filename + ".csv").exists():
            print("Loading {} data from {} to {}".format(filename, self.url, data_path))
            download(source_url=self.url,
                     dest_file_path=Path(data_path).joinpath(filename + ".csv"))
            mark_done(data_path)

        data = {filename: pd.read_csv(Path(data_path).joinpath(filename + ".csv"))}

        new_data = {'train': [],
                    'valid': [],
                    'test': []}

        limits = {'train': range(int(data[filename].shape[0] * 0.8)),
                  'valid': range(int(data[filename].shape[0] * 0.8),
                                 int(data[filename].shape[0] * 0.9)),
                  'test': range(int(data[filename].shape[0] * 0.9),
                                data[filename].shape[0])}

        for field in limits:
            for i in limits[field]:
                new_data[field].append((data[field].loc[i, 'text'],
                                        data[field].loc[i, "airline_sentiment"]))

        return new_data
