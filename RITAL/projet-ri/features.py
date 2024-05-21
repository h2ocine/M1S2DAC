from rank_bm25 import BM25Okapi
import utils
import textstat
import numpy as np
import math
from collections import Counter
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import string
import spacy
#from wikipedia2vec import Wikipedia2Vec
from sklearn.metrics.pairwise import cosine_similarity



# Load pre-trained Wikipedia2Vec model
#wiki2vec = Wikipedia2Vec.load('enwiki_20180420_500d.txt')
# Charger le modèle Word2Vec pré-entraîné
nltk.download('punkt')
model = api.load("word2vec-google-news-300")
nlp = spacy.load('en_core_web_sm')

#Length Number of terms in the sentence
def length(answer):
    return len(answer.split())

# ExactMatch Whether query is a substring
def check_exact_match(query, text):
    if query.lower() in text.lower():
        return 1  # Exact match found
    else:
        return 0  # No exact match found
    
#Overlap Fraction of query terms covered
def calculate_overlap_fraction(qrep, rep):
    overlap_fractions = []
    
    for qtext in qrep:
        qterms = set(qtext.lower().split())
        for _, text in rep:
            text = text.lower()
            tterms = set(text.split())
            overlap = len(qterms.intersection(tterms))
            fraction = overlap / len(qterms) if len(qterms) > 0 else 0
            overlap_fractions.append(fraction)
    
    return overlap_fractions


def overlap_syn_fraction(query, document):
    """
    Calcule la fraction des mots de la requête (ou leurs synonymes) présents dans le document.
    
    query : str : La requête (question)
    document : str : Le document (réponse candidate)
    
    return : float : La fraction des mots de la requête couverts.
    """
    query_tokens = utils.tokenize_text(query)
    document_tokens = utils.tokenize_text(document)
    print("frere")
    total_synonyms_count = 0
    overlap_count = 0
    print(f'len {query_tokens}')
    for token in query_tokens:
        synonyms = utils.get_synonyms(token)
        synonyms.add(token)  # Ajouter le mot lui-même à l'ensemble de ses synonymes
        total_synonyms_count += len(synonyms)

        for syn in synonyms:
            overlap_count += document_tokens.count(syn)
    
    if total_synonyms_count == 0:
        return 0.0
    print(f'len de overlaf {overlap_count}')
    print("frr")
    
    return overlap_count / len(document_tokens)



def get_entities(text):
    """
    Extracts named entities from a given text using spaCy.
    
    text : str : The text from which to extract entities
    
    return : set : A set of entity texts
    """
    doc = nlp(text)
    return set(ent.text for ent in doc.ents)

def tagme_overlap(query, document):
    """
    Calculates the fraction of named entities in the query that are also present in the document.
    
    query : str : The query (question)
    document : str : The document (candidate answer)
    
    return : float : The fraction of named entities in the query that are present in the document
    """
    query_entities = get_entities(query)
    print(f'query entitited {query_entities}')
    document_entities = get_entities(document)
    
    if not query_entities:
        return 0.0
    
    overlap = query_entities.intersection(document_entities)
    overlap_fraction = len(overlap) / len(query_entities)
    
    return overlap_fraction



def get_wikipedia2vec_vector(wiki2vec, text):
    """
    Get the Wikipedia2Vec vector for a given text.
    
    wiki2vec : Wikipedia2Vec : The Wikipedia2Vec model
    text : str : The text to vectorize
    
    return : array : The Wikipedia2Vec vector
    """
    words = text.split()
    vectors = [wiki2vec.get_word_vector(word) for word in words if word in wiki2vec.dictionary]
    return sum(vectors) / len(vectors) if vectors else None

def wikipedia2vec_similarity(wiki2vec, text1, text2):
    """
    Calculate the cosine similarity between two texts using their Wikipedia2Vec vectors.
    
    wiki2vec : Wikipedia2Vec : The Wikipedia2Vec model
    text1 : str : The first text
    text2 : str : The second text
    
    return : float : The cosine similarity between the two texts
    """
    vec1 = get_wikipedia2vec_vector(wiki2vec, text1)
    vec2 = get_wikipedia2vec_vector(wiki2vec, text2)
    if vec1 is None or vec2 is None:
        return 0.0,  
    return cosine_similarity([vec1], [vec2])[0][0]




def bm25_score(query, document, corpus, k1=1.5, b=0.75):
    """
    Calcule le score BM25 pour une paire de query-document.
    
    query : str : La requête (question)
    document : str : Le document (réponse candidate)
    corpus : list of str : La collection de tous les documents (corpus)
    k1 : float : Paramètre de saturation du terme (default 1.5)
    b : float : Paramètre de normalisation de la longueur du document (default 0.75)
    
    return : float : Le score BM25 du document pour la requête donnée
    """
    
    # Tokenisation de la query et du document
    query_terms = query.lower().split()
    document_terms = document.lower().split()
    
    # Longueur du document et longueur moyenne des documents du corpus
    len_d = len(document_terms)
    avgdl = sum(len(doc.split()) for doc in corpus) / len(corpus)
    
    # Calcul du nombre de documents dans le corpus
    N = len(corpus)
    
    # Calcul de la fréquence des termes dans le document
    freq_d = Counter(document_terms)
    
    # Calcul de la fréquence des documents pour chaque terme de la query
    df = {term: sum(1 for doc in corpus if term in doc.lower().split()) for term in query_terms}
    
    # Calcul du score BM25
    score = 0.0
    for term in query_terms:
        if term in freq_d:
            # Calcul du terme IDF
            idf = math.log(1 + (N - df[term] + 0.5) / (df[term] + 0.5))
            
            # Calcul du terme TF
            tf = freq_d[term] * (k1 + 1) / (freq_d[term] + k1 * (1 - b + b * len_d / avgdl))
            
            # Ajout du score du terme au score total
            score += idf * tf
            
    return score



def word2vec_similarity(query, document):
    query_terms = [word for word in query.lower().split() if word in model]
    document_terms = [word for word in document.lower().split() if word in model]
    
    if not query_terms or not document_terms:
        return 0.0
    
    query_vector = np.mean([model[word] for word in query_terms], axis=0)
    document_vector = np.mean([model[word] for word in document_terms], axis=0)
    
    return cosine_similarity([query_vector], [document_vector])[0][0]


import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Charger le modèle Word2Vec pré-entraîné
model = api.load("word2vec-google-news-300")

def get_word_vector(word):
    try:
        return model[word]
    except KeyError:
        return np.zeros(model.vector_size)

def vectorize_text(text):
    tokens = word_tokenize(text.lower())
    vectors = [get_word_vector(token) for token in tokens if token in model]
    return vectors

def ngrams(sequence, n):
    return [sequence[i:i + n] for i in range(len(sequence) - n + 1)]

def matched_ngram_similarity(question, answer, k, n):
    question_vectors = vectorize_text(question)
    answer_vectors = vectorize_text(answer)
    
    if len(question_vectors) < k or len(answer_vectors) < n:
        return 0.0

    max_similarity = 0.0
    question_grams = ngrams(question_vectors, k)
    answer_grams = ngrams(answer_vectors, n)
    
    for q_gram in question_grams:
        q_gram_sum = np.sum(q_gram, axis=0).reshape(1, -1)
        for a_gram in answer_grams:
            a_gram_sum = np.sum(a_gram, axis=0).reshape(1, -1)
            similarity = cosine_similarity(q_gram_sum, a_gram_sum)[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
    
    return max_similarity




#---------------------------------------------------------------------------------------------------
#---------------------------------------FONCTION-DE-LISIBILITE--------------------------------------
#---------------------------------------------------------------------------------------------------


def cpw(sentence):
    """
    Nombre moyen de caractères par mot dans une phrase.
    """
    words = sentence.split()  # Divise la phrase en mots.
    return np.mean([len(word) for word in words])  # Calcule la longueur moyenne des mots.

def spw(sentence):
    """
    Nombre moyen de syllabes par mot dans une phrase.
    """
    words = sentence.split()  # Divise la phrase en mots.
    return np.mean([textstat.syllable_count(word) for word in words])  # Calcule le nombre moyen de syllabes par mot.

def wps(sentence):
    """
    Nombre de mots par phrase.
    """
    return len(sentence.split())

def cwps(sentence):
    """
    Nombre de mots complexes par phrase (mots ayant plus de 2 syllabes).
    """
    words = sentence.split()  # Divise la phrase en mots.
    return len([word for word in words if textstat.syllable_count(word) > 2])  # Compte le nombre de mots complexes.

def cwr(sentence):
    """
    Fraction de mots complexes par rapport au nombre total de mots dans une phrase.
    """
    words = sentence.split()  # Divise la phrase en mots.
    return len([word for word in words if textstat.syllable_count(word) > 2]) / len(words)  # Calcule la fraction de mots complexes.

def lwps(sentence):
    """
    Nombre de mots longs par phrase (mots ayant plus de 7 caractères).
    """
    words = sentence.split()  # Divise la phrase en mots.
    return len([word for word in words if len(word) > 7])  # Compte le nombre de mots longs.

def lwr(sentence):
    """
    Fraction de mots longs par rapport au nombre total de mots dans une phrase.
    """
    words = sentence.split()  # Divise la phrase en mots.
    return len([word for word in words if len(word) > 7]) / len(words)  # Calcule la fraction de mots longs.


def dale_chall(sentence):
    """
    Score de lisibilité de Dale-Chall pour une phrase.
    """
    return textstat.dale_chall_readability_score(sentence)  # Calcule le score de lisibilité de Dale-Chall pour la phrase.
