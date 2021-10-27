import nmslib
from itertools import groupby
from collections import namedtuple
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from gensim.matutils import corpus2dense
from texts_processors import SimpleTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity


def texts_tokenizer(texts: []):
    """если захочется накручивать лингвистику, то сделать это тут"""
    tokenizer = SimpleTokenizerFast({})
    return tokenizer(texts)


def full_indexes_search(searched_ids: [], searched_texts: [], ids_in: [],
                        texts_in: [], min_score: float,
                        only_different_groups=True) -> [tuple]:

    """
    Function for searching duplicates in two texts collections with groups numbers.
    Algorithm with full indexing (all vectors are indexed each with each).
    searched_ids - IDs of texts to be found
    searched_texts - texts to be found
    ids - IDs of texts in which we will search
    texts - texts in which we will search
    min_score - similarity coefficient
    """

    texts = searched_texts + texts_in
    ids = searched_ids + ids_in

    texts_ids_dict = {num: item for num, item in enumerate(set(zip(texts, ids)))}

    lem_questions = texts_tokenizer([texts_ids_dict[x][0] for x in texts_ids_dict])
    dct = Dictionary(lem_questions)

    corpus = [dct.doc2bow(lm_tx) for lm_tx in lem_questions]
    dense_corpus = corpus2dense(corpus, num_terms=len(dct))
    np_corpus = dense_corpus.T

    index = nmslib.init()
    index.addDataPointBatch(np_corpus)
    index.createIndex({'post': 2}, print_progress=True)

    neighbours = index.knnQueryBatch(np_corpus, k=len(lem_questions), num_threads=8)

    min_distance = 1 - min_score
    neighbours_scores = [[x for x in zip(item[0], item[1]) if x[1] < min_distance] for item in neighbours]

    Duplicates = namedtuple("Duplicate", "searched_text, searched_id, similar_text, similar_text_id, score")
    results_indexes_items = []
    if only_different_groups:
        sorted_idx_groups_tuples = sorted([(i, texts_ids_dict[i][1]) for i in texts_ids_dict], key=lambda x: x[1])
        tx_ids_dict = {y: [i[0] for i in z] for y, z in groupby(sorted_idx_groups_tuples, key=lambda x: x[1])}

        for tx_num, similar_indexes in enumerate(neighbours_scores):
            delta_indexes = set([x[0] for x in similar_indexes]) - set(tx_ids_dict[texts_ids_dict[tx_num][1]])
            if delta_indexes:
                results_indexes_items += [(tx_num, x[0], 1.0 - x[1]) for x in similar_indexes
                                          if x[0] in delta_indexes]

    else:
        for tx_num, similar_indexes in enumerate(neighbours_scores):
            results_indexes_items += [(tx_num, x[0], 1.0 - x[1]) for x in similar_indexes]

    d = texts_ids_dict
    search_results = [Duplicates(d[idx1][0], d[idx1][1], d[idx2][0], d[idx2][1], score) for idx1, idx2, score in
                      results_indexes_items if d[idx1][1] in set(searched_ids) and d[idx2][1] in set(ids_in)]

    return sorted(search_results, key=lambda x: x[4], reverse=True)


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
    distances_arr = cosine_similarity(texts_corpus_csc, searched_texts_corpus_csc).T

    search_results = []
    Duplicates = namedtuple("Duplicate", "searched_text, searched_id, similar_text, similar_text_id, score")
    searched_texts_ids = zip(searched_texts, searched_ids)
    for srch_tx_ids, distances in zip(searched_texts_ids, distances_arr):
        initial_texts_ids = zip(texts, ids)
        search_results += [Duplicates(srch_tx_ids[0], srch_tx_ids[1], tx_ids[0], tx_ids[1], score) for
                           tx_ids, score in zip(initial_texts_ids, distances) if score >= min_score]

    return sorted(search_results, key=lambda x: x[4], reverse=True)
