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

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.file import read_json
from deeppavlov.core.models.component import Component
from deeppavlov.core.commands.infer import build_model_from_config

logger = get_logger(__name__)


@register('squad_wrapper')
class SquadWrapper(Component):
    """
    Load a SQLite database, read data batches and get docs content.
    """

    def __init__(self, squad_config_path, *args, **kwargs):
        self.squad_pipeline = build_model_from_config(read_json(squad_config_path))

    def __call__(self, question, contexts, *args, **kwargs):
        answers = []
        for c in contexts[0][0]:
            squad_answer = self.squad_pipeline([(c, question[0])])
            answers.append(squad_answer[0][0])
        return answers
