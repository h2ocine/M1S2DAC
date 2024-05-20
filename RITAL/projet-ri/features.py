from rank_bm25 import BM25Okapi
import utils
import textstat
import numpy as np


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

#Overlap synonym fractions 
def overlap_syn_fraction(query, document):
    """
    Calcule la fraction de synonymes des termes de la requête présents dans le document.
    
    query : str : La requête (question)
    document : str : Le document (réponse candidate)
    
    return : float : La fraction de synonymes couverts.
    """
    # Tokenisation de la requête et du document
    query_terms = utils.tokenize_text(query)
    document_terms = utils.tokenize_text(document)
    
    # Récupération des synonymes des termes de la requête
    query_synonyms = {term: utils.get_synonyms(term) for term in query_terms}
    
    # Compter le nombre de synonymes de la requête présents dans le document
    overlap_count = 0
    total_synonyms = sum(len(synonyms) for synonyms in query_synonyms.values())
    
    if total_synonyms == 0:
        return 0.0
    
    for term, synonyms in query_synonyms.items():
        if any(syn in document_terms for syn in synonyms):
            overlap_count += 1
    
    # Calcul de la fraction de synonymes couverts
    overlap_fraction = overlap_count / total_synonyms
    
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
