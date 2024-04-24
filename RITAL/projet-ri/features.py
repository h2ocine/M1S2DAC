from rank_bm25 import BM25Okapi





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
def overlap_syn_fraction(qrep, rep):
    """
    This function calculates the overlap synonym fraction between a query and a set of answers by finding synonyms of words in the query
    and comparing them with the synonyms of words in each answer.

    qrep : questions - list of String
    rep : answers - list of String
"""
    q_synonyms = {word: utils.get_synonyms(word) for word in utils.tokenize_text(qrep)}
    q_synonyms_lens = {word: len(synonyms) for word, synonyms in q_synonyms.items()}
    for text in rep:
        text_tokens = utils.tokenize_text(text)
        t_synonyms = set()
        for token in text_tokens:
            t_synonyms.update(utils.get_synonyms(token))
        overlap = sum(1 for syn in q_synonyms if syn in t_synonyms)
        fraction = overlap / sum(q_synonyms_lens.values()) if sum(q_synonyms_lens.values()) > 0 else 0
        yield fraction

# BM25 Score
def calculate_bm25_score(query, document):
    tokenized_query = query.lower().split()
    tokenized_document = document.lower().split()
    
    corpus = [tokenized_document]
    bm25 = BM25Okapi(corpus)
    
    #average_idf = sum(map(lambda k: float(bm25.idf[k]), bm25.idf.keys())) / len(bm25.idf.keys())
    
    score = bm25.get_scores(tokenized_query)
    
    # # Sort the answers based on BM25 scores in descending order
    # sorted_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    # sorted_answers = [corpus[i] for i in sorted_indices]
    return score[0]  # Aif only one document in the corpus