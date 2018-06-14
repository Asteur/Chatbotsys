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

# check version
import sys
assert sys.hexversion >= 0x3060000, 'Does not work in python3.5 or lower'

import deeppavlov.dataset_iterators.sqlite_iterator
import deeppavlov.models.tokenizers.spacy_tokenizer
import deeppavlov.models.tokenizers.ru_tokenizer
import deeppavlov.models.supplementary.query_paragraph_tuplifier
import deeppavlov.models.supplementary.rank_doc_score_tuplifier
import deeppavlov.vocabs.wiki_sqlite
import deeppavlov.skills.odqa.tfidf_ranker
import deeppavlov.core.common.log
import deeppavlov.download
