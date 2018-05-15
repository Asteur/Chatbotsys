#%% imports
from deeppavlov.core.commands.utils import set_deeppavlov_root, expand_path
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.models.preprocessors.squad_preprocessor import SquadPreprocessor, SquadAnsPreprocessor, \
    SquadVocabEmbedder, SquadAnsPostprocessor
from deeppavlov.models.squad.squad import SquadModel
from deeppavlov.models.tokenizers.spacy_tokenizer import StreamSpacyTokenizer
from deeppavlov.models.vectorizers.hashing_tfidf_vectorizer import HashingTfIdfVectorizer
from deeppavlov.skills.odqa.ranker import TfidfRanker

#%% init
from deeppavlov.vocabs.wiki_sqlite import WikiSQLiteVocab

set_deeppavlov_root({})


#%% init chainer
chainer = Chainer(in_x='question_raw', out_params='ans_predicted')

#%% init and add ranker
tokenizer = StreamSpacyTokenizer(lemmas=True, ngram_range=[1, 2])
enwiki_matrix_path = "odqa/enwiki_tfidf_matrix.npz"
vectorizer = HashingTfIdfVectorizer(save_path=enwiki_matrix_path, load_path=enwiki_matrix_path, tokenizer=tokenizer)

chainer.append(TfidfRanker(vectorizer=vectorizer), out_params=["doc_ids", "doc_scores"])

chainer.append(WikiSQLiteVocab(data_dir='odqa', shuffle=False,
                               data_url='http://lnsigo.mipt.ru/export/datasets/wikipedia/enwiki.db'),
               in_x='doc_ids', out_params='context_raw')

#%% squad
limits = {
    'context_limit': 400,
    'question_limit': 150,
    'char_limit': 16
}

chainer.append(SquadPreprocessor(**limits), in_x=["context_raw", "question_raw"],
               out_params=["context", "context_tokens", "context_chars", "c_r2p", "c_p2r", "question",
                           "question_tokens", "question_chars", "spans"])

vocab_emb_path = 'squad_model/emb/vocab_embedder.pckl'
vocab_embedder = SquadVocabEmbedder(level='token', emb_folder='embeddings/',
                                    emb_url='http://lnsigo.mipt.ru/export/embeddings/wiki-news-300d-1M.vec',
                                    save_path=vocab_emb_path, load_path=vocab_emb_path, **limits)
chainer.append(vocab_embedder, in_x=["context_tokens", "question_tokens"],
               out_params=["context_tokens_idxs", "question_tokens_idxs"])

char_vocab_emb_path = 'squad_model/emb/char_vocab_embedder.pckl'
char_vocab_embedder = SquadVocabEmbedder(level='char', emb_folder='embeddings/',
                                         emb_url='http://lnsigo.mipt.ru/export/embeddings/wiki-news-300d-1M-char.vec',
                                         save_path=char_vocab_emb_path, load_path=char_vocab_emb_path, **limits)
chainer.append(char_vocab_embedder, in_x=["context_chars", "question_chars"],
               out_params=["context_chars_idxs", "question_chars_idxs"])

model_path = 'squad_model/model'
squad_model = SquadModel(word_emb=vocab_embedder.emb_mat, char_emb=char_vocab_embedder.emb_mat, **limits,
                         train_char_emb=True, char_hidden_size=100, encoder_hidden_size=75, attention_hidden_size=75,
                         learning_rate=0.5, min_learning_rate=0.001, learning_rate_patience=5, keep_prob=0.7,
                         grad_clip=5.0, weight_decay=1.0, save_path=model_path, load_path=model_path)
chainer.append(squad_model,
               in_x=["context_tokens_idxs", "context_chars_idxs", "question_tokens_idxs", "question_chars_idxs"],
               out_params=["ans_start_predicted", "ans_end_predicted"])

chainer.append(SquadAnsPostprocessor(),
               in_x=["ans_start_predicted", "ans_end_predicted", "context_raw", "c_p2r", "spans"],
               out_params=["ans_predicted", "ans_start_predicted", "ans_end_predicted"])

#%% interact
while True:
    args = [input('question_raw::')]
    if args[-1] in {'exit', 'stop', 'quit', 'q'}:
        break

    print('>>', *chainer(args))
