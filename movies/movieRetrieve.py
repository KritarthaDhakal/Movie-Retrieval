import re
import os
import math
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict, Counter

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def load_documents(folder_path):
    docs = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                docs[filename] = file.read()
    return docs


def load_queries(query_path):
    with open(query_path, 'r') as file:
        queries = file.readlines()
    queries = [query.strip() for query in queries]
    return queries


def text_preprocessing(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return cleaned_tokens




'''
    For the Vector Space Model with Cosine Similarity

                                                        '''


def calculate_stats_vsm(documents):
    vocab = set()
    N = len(documents)
    tf_documents = {}
    df = Counter()

    # df
    for doc_name, doc in documents.items():
        tokens = text_preprocessing(doc)
        unique_tokens = set(tokens)
        df.update(unique_tokens)

        # tf
        total_terms = len(tokens)
        term_count = Counter(tokens)
        term_frequency = {term: count/total_terms for term, count in term_count.items()}
        tf_documents[doc_name] = term_frequency
        vocab.update(tokens)

    # idf
    idf = {term: math.log(N / (1 + df[term])) for term in df}

    # tf-idf
    tfidf_documents = {}
    for doc_name, tf_dict in tf_documents.items():
        tfidf = {term: tf_value * idf.get(term, 0) for term, tf_value in tf_dict.items()}
        tfidf_documents[doc_name] = tfidf

    return idf, tfidf_documents


def cosine_similarity(vec1, vec2):
    all_terms = sorted(set(vec1) | set(vec2))
    vec1_list = np.array([vec1.get(term, 0) for term in all_terms])
    vec2_list = np.array([vec2.get(term, 0) for term in all_terms])
    dot_product = np.dot(vec1_list, vec2_list)
    norm_vec1 = np.linalg.norm(vec1_list)
    norm_vec2 = np.linalg.norm(vec2_list)
    return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 != 0 and norm_vec2 != 0 else 0


def ranking_function_vsm(documents, queries):

    idf, tfidf_documents = calculate_stats_vsm(documents)
    
    ranked_results = {}
    for query_num, query in enumerate(queries, start=1): 
        query_tokens = text_preprocessing(query)
        query_tf = Counter(query_tokens)
        query_total_terms = len(query_tokens)
        query_tf_normalized = {term: count/query_total_terms for term, count in query_tf.items()}
        query_tfidf = {term: query_tf_normalized[term] * idf.get(term, 0) for term in query_tf_normalized}

        scores = {}
        for doc_name, doc_tfidf in tfidf_documents.items():
            score = cosine_similarity(doc_tfidf, query_tfidf)
            scores[doc_name] = score

        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_results[query_num] = [doc_name for doc_name, score in ranked_docs]

    return ranked_results



'''
    For the Binary Independence Model

                                        '''


def calculate_stats_bim(documents):
    
    tfs = {}
    dfs = defaultdict(int)
    for doc_id, content in documents.items():
        terms = text_preprocessing(content)

        tfs[doc_id] = defaultdict(int)
        for term in terms:
            tfs[doc_id][term] += 1

        for term in set(terms):
            dfs[term] += 1

    return tfs, dfs


def calculate_bim_scores(query, tfs, dfs, doc_count):
    scores = {}
    for doc_id in tfs:
        score = 1.0
        for term in query:
            tf = tfs[doc_id].get(term, 0)
            df = dfs.get(term, 0)
            p_term_given_doc = (tf + 1) / (sum(tfs[doc_id].values()) + len(dfs))
            p_term_given_corpus = (df + 1) / (doc_count + len(dfs))
            score *= (p_term_given_doc / p_term_given_corpus)
        scores[doc_id] = score
    return scores


def ranking_function_bim(docs, queries):

    doc_count = len(docs)
    tfs, dfs = calculate_stats_bim(docs)

    ranked_results = {}
    for query_num, query in enumerate(queries, start=1):
        query_terms = text_preprocessing(query)
        scores = calculate_bim_scores(query_terms, tfs, dfs, doc_count)
        
        ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ranked_results[query_num] = [doc_name for doc_name, score in ranked_docs]

    return ranked_results



'''
    For the Unigram Model with Laplace Smoothing

                                        '''


def calculate_stats_unigram(documents):
    tfs = {}
    doc_lengths = {}
    vocab = set()

    for doc_id, content in documents.items():
        tokens = text_preprocessing(content)
        tf = Counter(tokens)
        tfs[doc_id] = tf
        doc_lengths[doc_id] = len(tokens)
        vocab.update(tokens)

    vocab_size = len(vocab)
    return tfs, doc_lengths, vocab_size



def create_unigram_model(tfs, doc_lengths, vocab_size):
    doc_unigrams = {}
    
    for doc_id, tf in tfs.items():
        doc_length = doc_lengths[doc_id]
        unigram_model = defaultdict(float)

        for term, freq in tf.items():
            unigram_model[term] = (freq + 1) / (doc_length + vocab_size)

        unigram_model['<UNK>'] = 1 / (doc_length + vocab_size)

        doc_unigrams[doc_id] = unigram_model
        
    return doc_unigrams



def score_documents_unigram(query, doc_unigrams):
    query_terms = text_preprocessing(query)
    scores = {}

    for doc_id, unigram_model in doc_unigrams.items():
        log_likelihood = 0

        for term in query_terms:
            prob = unigram_model.get(term, unigram_model['<UNK>'])
            log_likelihood += math.log(prob)

        scores[doc_id] = log_likelihood
    return scores



def ranking_function_unigram(docs, queries):
    
    tfs, doc_lengths, vocab_size = calculate_stats_unigram(docs)
    
    doc_unigrams = create_unigram_model(tfs, doc_lengths, vocab_size)

    ranked_results = {}
    for query_num, query in enumerate(queries, start=1):
        doc_scores = score_documents_unigram(query, doc_unigrams)
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_results[query_num] =  [doc_name for doc_name, score in ranked_docs]

    return ranked_results



folder_path = "movies\dataset"
query_file_path = "movies\queries.txt"
    
documents = load_documents(folder_path)
queries = load_queries(query_file_path)

ranked_results_vsm = ranking_function_vsm(documents, queries)
ranked_results_bim = ranking_function_bim(documents, queries)
ranked_results_unigram = ranking_function_unigram(documents, queries)