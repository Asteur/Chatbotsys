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

from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.core.commands.infer import interact_model, predict_on_stream
from deeppavlov.core.common.log import get_logger
from deeppavlov.download import download_resources
from utils.telegram_utils.telegram_ui import interact_model_by_telegram
from utils.server_utils.server import start_model_server


log = get_logger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("mode", help="select a mode, train or interact", type=str,
                    choices={'train', 'interact', 'predict', 'interactbot', 'riseapi', 'download'})
parser.add_argument("config_path", help="path to a pipeline json config", type=str)
parser.add_argument("-t", "--token", help="telegram bot token", type=str)
parser.add_argument("-b", "--batch-size", dest="batch_size", default=1, help="inference batch size", type=int)
parser.add_argument("-f", "--input-file", dest="file_path", default=None, help="Path to the input file", type=str)
parser.add_argument("-d", "--download", action="store_true", help="download model components")


def main():
    args = parser.parse_args()
    pipeline_config_path = args.config_path

    token = args.token or os.getenv('TELEGRAM_TOKEN')

    if args.download or args.mode == 'download':
        download_resources(Path(pipeline_config_path).resolve())

    if args.mode == 'train':
        train_model_from_config(pipeline_config_path)
    elif args.mode == 'interact':
        interact_model(pipeline_config_path)
    elif args.mode == 'interactbot':
        if not token:
            log.error('Token required: initiate -t param or TELEGRAM_BOT env var with Telegram bot token')
        else:
            interact_model_by_telegram(pipeline_config_path, token)
    elif args.mode == 'riseapi':
        start_model_server(pipeline_config_path)
    elif args.mode == 'predict':
        predict_on_stream(pipeline_config_path, args.batch_size, args.file_path)


if __name__ == "__main__":
    main()
