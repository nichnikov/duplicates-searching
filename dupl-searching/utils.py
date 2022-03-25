from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from texts_processors import SimpleTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity


def texts_tokenizer(texts: []):
    """если захочется накручивать лингвистику, то сделать это тут"""
    tokenizer = SimpleTokenizerFast({})
    return tokenizer(texts)


def duplicates_search_func(searched_ids, searched_texts, ids, texts, min_score):
    """Function for searching duplicates in two texts collections with groups numbers.
    searched_ids - IDs of texts to be found
    searched_texts - texts to be found
    ids - IDs of texts in which we will search
    texts - texts in which we will search
    min_score - similarity coefficient
    """

    lem_texts = texts_tokenizer(texts)
    lem_searched_texts = texts_tokenizer(searched_texts)

    """Vectorization"""
    dct = Dictionary(lem_searched_texts + lem_texts)

    texts_corpus_csc = corpus2csc([dct.doc2bow(lm_tx) for lm_tx in lem_texts], num_terms=len(dct))
    texts_corpus_csc = texts_corpus_csc.T

    searched_texts_corpus_csc = corpus2csc([dct.doc2bow(lm_tx) for lm_tx in lem_searched_texts], num_terms=len(dct))
    searched_texts_corpus_csc = searched_texts_corpus_csc.T

    """Duplicates Searching"""
    # distances_arr = cosine_similarity(texts_corpus_csc, searched_texts_corpus_csc, dense_output=False).T
    distances_arr = cosine_similarity(texts_corpus_csc, searched_texts_corpus_csc).T

    search_results = []
    # Duplicates = namedtuple("searched_text, searched_id, similar_text, similar_text_id, score")
    searched_texts_ids = zip(searched_texts, searched_ids)
    for srch_tx_ids, distances in zip(searched_texts_ids, distances_arr):
        initial_texts_ids = zip(texts, ids)
        search_results += [(srch_tx_ids[0], srch_tx_ids[1], tx_ids[0], tx_ids[1], score) for
                           tx_ids, score in zip(initial_texts_ids, distances) if score >= min_score]

    return sorted(search_results, key=lambda x: x[4], reverse=True)
