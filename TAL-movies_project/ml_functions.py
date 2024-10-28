from sklearn import (
    linear_model, 
    ensemble,
    tree,
    decomposition, 
    naive_bayes, 
    neural_network,
    svm,
    metrics,
    preprocessing, 
    model_selection, 
    pipeline,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, auc, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

import pandas as pd
import numpy as np



red_code = '\033[91m'
blue_code = '\033[94m'
green_code = '\033[92m'
yellow_code = '\033[93m'
reset_code = '\033[0m'


# ------------------------------------------------------------------- ------------------------------------------------------------------- -------------------------------------------------------------------
# ------------------------------------------------------------------- ------------------------------------------------------------------- -------------------------------------------------------------------
# ------------------------------------------------------------------- ------------------------------------------------------------------- -------------------------------------------------------------------


# Fonction d'apprentissage
def analyze(data, vectorizer, model):
    """
    Effectue une analyse en utilisant le modèle et le vectorizer spécifié.
    """
    # Diviser les données en ensembles d'entraînement et de test
    X_text_train, X_text_test, y_train, y_test = model_selection.train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

    # Transformation des données d'entraînement en utilisant le vectoriseur
    X_train = vectorizer.fit_transform(X_text_train)
    # Transformation des données de test en utilisant le même vectoriseur
    X_test = vectorizer.transform(X_text_test)

    # Entraîner le modèle sur les données d'entraînement
    model.fit(X_train, y_train)

    # Prédire les étiquettes des données de test
    y_pred = model.predict(X_test)

    # Prédire les probabilités des classes positives pour les données de test
    # Prédire les probabilités des classes positives pour les données de test
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Utiliser la décision de fonction de décision si le modèle ne prend pas en charge predict_proba
        y_prob = model.decision_function(X_test)

    # Calcul des métriques de performance
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = metrics.auc(fpr, tpr)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # Affichage du rapport de classification
    report = metrics.classification_report(y_test, y_pred)
    # print(report)
    
    return acc, f1, auc


# Fonction d'apprentissage régression logistique
     
def logistic_regression(X_train, X_test, y_train, y_test):
    model = clf = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000)
    # Entraîner le modèle sur les données d'entraînement
    model.fit(X_train, y_train)

    # Prédire les étiquettes des données de test
    y_pred = model.predict(X_test)

    # Prédire les probabilités des classes positives pour les données de test
    # Prédire les probabilités des classes positives pour les données de test
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Utiliser la décision de fonction de décision si le modèle ne prend pas en charge predict_proba
        y_prob = model.decision_function(X_test)

    # Calcul des métriques de performance
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = metrics.auc(fpr, tpr)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # Affichage du rapport de classification
    report = metrics.classification_report(y_test, y_pred)
    # print(report)

    # print(f'{green_code}Accuracy :\t{acc}{reset_code}')
    # print(f'{green_code}F1 score :\t{f1}{reset_code}')
    # print(f'{green_code}AUC :\t\t{auc}{reset_code}')
    return acc, f1, auc


# Fonctions d'apprentissage par modèles
    
def logistic_regression_analyze(data, vectorizer):
    # Initialiser un modèle de régression logistique
    clf = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000)

    # Utiliser la fonction "analyze" avec le modèle de régression logistique
    return analyze(data, vectorizer, clf)

# -------------------------------------------------------------------
# -------------------------------------------------------------------

def svm_analyze(data, vectorizer):
    # Initialiser un modèle SVM
    clf = SVC()

    # Utiliser la fonction "analyze" avec le modèle SVM
    return analyze(data, vectorizer, clf)

# -------------------------------------------------------------------
# -------------------------------------------------------------------

def decision_tree_analyze(data, vectorizer):
    # Initialiser un modèle d'arbre de décision
    clf = DecisionTreeClassifier()

    # Utiliser la fonction "analyze" avec le modèle d'arbre de décision
    return analyze(data, vectorizer, clf)

# -------------------------------------------------------------------
# -------------------------------------------------------------------

def random_forest_analyze(data, vectorizer):
    # Initialiser un modèle de forêt aléatoire
    clf = RandomForestClassifier()

    # Utiliser la fonction "analyze" avec le modèle d'arbre de décision
    return analyze(data, vectorizer, clf)

# -------------------------------------------------------------------
# -------------------------------------------------------------------

# Fonctions d'apprentissage par vectorizer

def count_analyze(data, analyze_function, **count_vectorizer_args):
    vectorizer = CountVectorizer(**count_vectorizer_args)
    return analyze_function(data, vectorizer)

def tfidf_analyze(data, analyze_function, **tfidf_vectorizer_args):
    vectorizer = TfidfVectorizer(**tfidf_vectorizer_args, sublinear_tf =  True)
    return analyze_function(data, vectorizer)

def hashing_analyze(data, analyze_function, **hashing_vectorizer_args):
    vectorizer = HashingVectorizer(**hashing_vectorizer_args)
    return analyze_function(data, vectorizer)

# -------------------------------------------------------------------
# -------------------------------------------------------------------

# Fonctions d'evaluations : 

def evaluation(data_set_df, analyze_function, **vectorizer_args):
    # print(f'{yellow_code}count{reset_code}')
    count_acc, count_f1, count_auc = count_analyze(data_set_df, analyze_function, **vectorizer_args)

    # print(f'\n{yellow_code}tfidf{reset_code}')
    tfidf_acc, tfidf_f1, tfidf_auc = tfidf_analyze(data_set_df, analyze_function, **vectorizer_args)

    return (count_acc, count_f1, count_auc), (tfidf_acc, tfidf_f1, tfidf_auc)


def all_evaluations(data_set_df, analyze_function):
    results = []
    
    evaluation_types = [
        ('Unigram', 1, (1,1), None),
        ('Bigram', 1, (1, 2), None),
        ('Trigram', 1, (1, 3), None),
        ('Unigram Stop words', 1, (1,1), 'english'),
        ('Unigram + Stop word', 1, (1, 2), 'english'),
        ('Bigram + Stop word', 1, (1, 2), 'english'),
        ('Trigram + Stop word', 1, (1, 3), 'english'),

        
        ('Reduction du vocabulaire Unigram', 0.7, (1,1), None),
        ('Reduction du vocabulaire Bigram', 0.7, (1, 2), None),
        ('Reduction du vocabulaire Trigram', 0.7, (1, 3), None),
        ('Reduction du vocabulaire Unigram Stop words', 0.7, (1,1), 'english'),
        ('Reduction du vocabulaire Unigram + Stop word', 0.7, (1, 2), 'english'),
        ('Reduction du vocabulaire Bigram + Stop word', 0.7, (1, 2), 'english'),
        ('Reduction du vocabulaire Trigram + Stop word', 0.7, (1, 3), 'english'),
    ]

    for eval_type, vocab_size, ngram_range, stop_words in evaluation_types:
        print(f'Entrainement et évaluation pour {blue_code}{eval_type}{reset_code}')
        (count_acc, count_f1, count_auc), (tfidf_acc, tfidf_f1, tfidf_auc) = evaluation(data_set_df, analyze_function, max_df=vocab_size, ngram_range=ngram_range, stop_words=stop_words)
        results.append({'Type': eval_type, 'Model': 'Count', 'Accuracy': count_acc, 'F1 Score': count_f1, 'AUC': count_auc})
        results.append({'Type': eval_type, 'Model': 'TF-IDF', 'Accuracy': tfidf_acc, 'F1 Score': tfidf_f1, 'AUC': tfidf_auc})

    results_df = pd.DataFrame(results)
    return results_df
