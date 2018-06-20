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
from typing import List

from nltk.tokenize import sent_tokenize

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register


@register("sentence_splitter")
class SentenceSplitter(Component):
    """
    Split a list of documents to a batches of list of sentences.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch: List[List[str]], *args, **kwargs) -> List[List[str]]:
        batch_sentences = []
        for docs in batch:
            if isinstance(docs, list):
                instance_sentences = []
                for doc in docs:
                    instance_sentences += sent_tokenize(doc)
                batch_sentences.append(instance_sentences)
            else:
                instance_sentences = sent_tokenize(docs)
                batch_sentences.append(instance_sentences)

        to_save = batch_sentences[0]
        # with open('en_drones_sentences.txt', 'w') as fout:
        #     fout.write("\n".join(to_save))
        return batch_sentences
