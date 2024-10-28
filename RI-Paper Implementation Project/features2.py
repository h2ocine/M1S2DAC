from rank_bm25 import BM25Okapi
import utils
import textstat
import numpy as np
import math
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import string
import spacy
#from wikipedia2vec import Wikipedia2Vec
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize

# Load pre-trained Wikipedia2Vec model
#wiki2vec = Wikipedia2Vec.load('enwiki_20180420_500d.txt')
nlp = spacy.load('en_core_web_sm')



#---------------------------------------------------------------------------------------------------
#---------------------------------------FONCTION-DE-LISIBILITE--------------------------------------
#---------------------------------------------------------------------------------------------------


def cpw(sentence):
    """
    Nombre moyen de caractères par mot dans une phrase.
    """
    words = sentence.split()  # Divise la phrase en mots.
    if not words:
        return 0
    return np.mean([len(word) for word in words])  # Calcule la longueur moyenne des mots.

def spw(sentence):
    """
    Nombre moyen de syllabes par mot dans une phrase.
    """
    words = sentence.split()  # Divise la phrase en mots.
    if not words:
        return 0
    return np.mean([textstat.syllable_count(word) for word in words])  # Calcule le nombre moyen de syllabes par mot.

def wps(sentence):
    """
    Nombre de mots par phrase.
    """
    return len(sentence.split())  # Compte le nombre de mots dans la phrase.

def cwps(sentence):
    """
    Nombre de mots complexes par phrase (mots ayant plus de 2 syllabes).
    """
    words = sentence.split()  # Divise la phrase en mots.
    if not words:
        return 0
    return len([word for word in words if textstat.syllable_count(word) > 2])  # Compte le nombre de mots complexes.

def cwr(sentence):
    """
    Fraction de mots complexes par rapport au nombre total de mots dans une phrase.
    """
    words = sentence.split()  # Divise la phrase en mots.
    if not words:
        return 0
    return len([word for word in words if textstat.syllable_count(word) > 2]) / len(words)  # Calcule la fraction de mots complexes.

def lwps(sentence):
    """
    Nombre de mots longs par phrase (mots ayant plus de 7 caractères).
    """
    words = sentence.split()  # Divise la phrase en mots.
    if not words:
        return 0
    return len([word for word in words if len(word) > 7])  # Compte le nombre de mots longs.

def lwr(sentence):
    """
    Fraction de mots longs par rapport au nombre total de mots dans une phrase.
    """
    words = sentence.split()  # Divise la phrase en mots.
    if not words:
        return 0
    return len([word for word in words if len(word) > 7]) / len(words)  # Calcule la fraction de mots longs.

def dale_chall(sentence):
    """
    Score de lisibilité de Dale-Chall pour une phrase.
    """
    if not sentence.strip():
        return 0
    return textstat.dale_chall_readability_score(sentence)  # Calcule le score de lisibilité de Dale-Chall pour la phrase.




#---------------------------------------------------------------------------------------------------
#--------------------------------- Lexical and semantic matching -----------------------------------
#---------------------------------------------------------------------------------------------------


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
def overlap(qrep, rep):
    """
    Calcule la fraction des termes de la requête présents dans la réponse.
    
    qrep : str : La requête (question)
    rep : str : La réponse (document)
    
    return : float : La fraction des termes de la requête couverts par la réponse.
    """
    rep = rep.lower()
    mots_question = qrep.lower().split()
    mots_reponse = rep.split()
    
    overlap = 0

    for mot_question in mots_question:
        if mot_question in mots_reponse:
            overlap += mots_reponse.count(mot_question)

    return overlap / len(mots_reponse) if len(mots_reponse) > 0 else 0.0
    

def overlap_syn_fraction(query, document):
    """
    Calcule la fraction des mots de la requête (ou leurs synonymes) présents dans le document.
    
    query : str : La requête (question)
    document : str : Le document (réponse candidate)
    
    return : float : La fraction des mots de la requête couverts.
    """
    # Tokeniser les textes
    query_tokens = utils.tokenize_text(query)
    document_tokens = utils.tokenize_text(document)
    
    total_synonyms_count = 0
    overlap_count = 0
    
    for token in query_tokens:
        # Obtenir les synonymes du token
        synonyms = utils.get_synonyms(token)
        synonyms.add(token)  # Ajouter le mot lui-même à l'ensemble de ses synonymes
        total_synonyms_count += len(synonyms)
        
        for syn in synonyms:
            overlap_count += document_tokens.count(syn)
    
    if total_synonyms_count == 0:
        return 0.0
    
    return overlap_count / len(document_tokens) if len(document_tokens) > 0 else 0.0


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
    document_entities = get_entities(document)
    
    if not query_entities:
        return 0.0
    
    overlap = query_entities.intersection(document_entities)
    overlap_fraction = len(overlap) / len(query_entities)
    
    return overlap_fraction

    
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







def word2vec_similarity(query, document, model):
    query_terms = [word for word in query.lower().split() if word in model]
    document_terms = [word for word in document.lower().split() if word in model]
    
    if not query_terms or not document_terms:
        return 0.0
    
    query_vector = np.mean([model[word] for word in query_terms], axis=0)
    document_vector = np.mean([model[word] for word in document_terms], axis=0)
    
    return cosine_similarity([query_vector], [document_vector])[0][0]














###########

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
        return 0.0
    return cosine_similarity([vec1], [vec2])[0][0]


