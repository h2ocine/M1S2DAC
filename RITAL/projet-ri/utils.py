import re
import unicodedata
import string
import nltk
import spacy
import os.path
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

nltk.download('wordnet')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

def preprocess(text, lemma = False):
    """
    Transforms text to remove unwanted bits.
    """
    
    # Characters suppression 
    
    # Non normalized char suppression 
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    
    # Lowercase transformation
    text = text.lower()

    # Punctuation suppression
    translation_table = str.maketrans("", "", string.punctuation + '\n\r\t')
    text = text.translate(translation_table)
    
    # Digits suppression
    text = re.sub(r'\d', '', text)
    # Use the sub function to replace double spaces with single spaces
    text = re.sub(r'\s+', ' ', text)
 
    if lemma:
        lemmatised_text = nlp(text)
        text = [str(word.lemma_) for word in lemmatised_text]
        text = ' '.join(text)
    
    return text


def get_synonyms(word):
    """
    get synonyms of a word
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

def tokenize_text(text):
    """ 
    Tokenize the text 
    """
    #stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    #tokens = [token for token in tokens if token not in stop_words]
    return tokens


