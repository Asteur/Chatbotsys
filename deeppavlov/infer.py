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

from pathlib import Path
import sys
from datetime import datetime, timedelta

p = (Path(__file__) / ".." / "..").resolve()
sys.path.append(str(p))

from deeppavlov.core.commands.train import build_model_from_config
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.utils import get_project_root

log = get_logger(__name__)

CONFIG_PATH = str(get_project_root()) + '/deeppavlov/configs/odqa/generic_ranker.json'

PERIOD = timedelta(minutes=5)
NEXT_TIME = datetime.now() + PERIOD
MINUTES = 0


def build_chainer(config_path):
    config = read_json(config_path)
    chainer = build_model_from_config(config)
    return chainer


def check_data_changed():
    global NEXT_TIME
    global MINUTES
    if NEXT_TIME <= datetime.now():
        MINUTES += 1
        NEXT_TIME += PERIOD
        return True
    else:
        NEXT_TIME = datetime.now() + PERIOD
        MINUTES = 0
        return False


def run(chainer):
    while True:
        if check_data_changed():
            chainer = build_chainer(CONFIG_PATH)
        try:
            query = input("Question: ")
            context = chainer([query.strip()])[0][0]
            print(context)
        except Exception:
            raise


def main(chainer):
    try:
        run(chainer)
    except Exception:
        run(chainer)


if __name__ == "__main__":
    model = build_chainer(CONFIG_PATH)
    main(model)
