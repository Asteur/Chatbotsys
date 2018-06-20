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

import argparse
from pathlib import Path
import sys
import os

p = (Path(__file__) / ".." / "..").resolve()
sys.path.append(str(p))

from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.commands.utils import get_project_root
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.log import get_logger

log = get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("input_path", help="Path to folder with data files.", type=str)
parser.add_argument("output_path", help="Path to a folder with trained data.", type=str)
args = parser.parse_args()
input_path = Path(args.input_path).resolve()
output_path = Path(args.output_path).resolve()

CONFIG_PATH = str(get_project_root()) + '/deeppavlov/configs/odqa/ru_ranker.json'
config = read_json(CONFIG_PATH)
config['dataset_reader']['data_path'] = input_path
config['dataset_reader']['save_path'] = config['dataset_iterator']['load_path'] = \
config['chainer']['pipe'][1]['load_path'] = os.path.join(output_path, 'data.db')
config['chainer']['pipe']['vectorizer']['save_path'] = config['chainer']['pipe']['vectorizer'][
    'load_path'] = os.path.join(output_path, 'tfidf.npz')

train_evaluate_model_from_config(config, pass_config=True)
