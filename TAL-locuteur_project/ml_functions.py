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

# Gaussian smoothing function
def countValue(l):
    d = {-1:0, 1:0}
    for v in l:
        # d[v] = d.get(v, 0) + 1
        d[v] += 1
    return d

def gaussian_smoothing(y, size=1):
    """
    Convolue sur la liste et regarde le nombre de voisin dans la fenêtre centré sur le point.
    Attribue simple la classe la plus présente dans la fenêtre au point centré.
    """
    new_y = y.copy()
    for i in range(size, len(y) - size):
        window = y[i - size : i + size + 1]
        d = countValue(window)
        if d[-1] < d[1]:
            new_y[i] = 1
        elif d[-1] > d[1]:
            new_y[i] = -1
        else:
            pass
    return new_y

# -------------------------------------------------------------------
# -------------------------------------------------------------------

# Fonction d'apprentissage
def analyze(data, vectorizer, model):
    """
    Effectue une analyse en utilisant le modèle et le vectorizer spécifié.
    """
    # Diviser les données en ensembles d'entraînement et de test
    X_text_train, X_text_test, y_train, y_test = model_selection.train_test_split(data['text'], data['label'], test_size=0.2, random_state=24, shuffle=False)

    X_text_train, _, y_train, _ = model_selection.train_test_split(X_text_train, y_train, test_size=0.000000001, random_state=0)

    # Transformation des données d'entraînement en utilisant le vectoriseur
    X_train = vectorizer.fit_transform(X_text_train)
    # Transformation des données de test en utilisant le même vectoriseur
    X_test = vectorizer.transform(X_text_test)

    # Entraîner le modèle sur les données d'entraînement
    model.fit(X_train, y_train)

    # Prédire les étiquettes des données de test
    y_pred = model.predict(X_test)

    # Appliquer le gaussian smoothing 
    window_gs = 2
    y_pred = gaussian_smoothing(y_pred, window_gs)

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

    # Calcul des scores d'exactitude pour chaque classe
    unique_classes = set(y_test)
    acc_per_class = {}
    for cls in unique_classes:
        cls_indices = (y_test == cls)
        cls_acc = accuracy_score(y_test[cls_indices], y_pred[cls_indices])
        acc_per_class[cls] = cls_acc

    # Affichage du rapport de classification
    report = metrics.classification_report(y_test, y_pred,output_dict=True, zero_division=1)
    # print(report)
    return acc, f1, auc, (report["-1"]["f1-score"],report["1"]["f1-score"])


# Fonction d'apprentissage régression logistique
     
def logistic_regression(X_train, X_test, y_train, y_test):
    model = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced')
    # Entraîner le modèle sur les données d'entraînement
    model.fit(X_train, y_train)

    # Prédire les étiquettes des données de test
    y_pred = model.predict(X_test)

    # Appliquer le gaussian smoothing 
    window_gs = 2
    y_pred = gaussian_smoothing(y_pred, window_gs)

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

    # Calcul des scores d'exactitude pour chaque classe
    acc_per_class = accuracy_score(y_test, y_pred, normalize=False)

    # Affichage du rapport de classification
    report = metrics.classification_report(y_test, y_pred,output_dict=True, zero_division=1)
    # print(report)

    return acc, f1, auc, (report["-1"]["f1-score"],report["1"]["f1-score"])


# Fonctions d'apprentissage par modèles

def logistic_regression_analyze(data, vectorizer):
    # Initialiser un modèle de régression logistique
    clf = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced')

    # Utiliser la fonction "analyze" avec le modèle de régression logistique
    return analyze(data, vectorizer, clf)
# -------------------------------------------------------------------
# -------------------------------------------------------------------

def svm_analyze(data, vectorizer):
    # Initialiser un modèle SVM
    clf = SVC(class_weight='balanced')

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
    count_acc, count_f1, count_auc, cout_acc_per_class = count_analyze(data_set_df, analyze_function, **vectorizer_args)

    # print(f'\n{yellow_code}tfidf{reset_code}')
    tfidf_acc, tfidf_f1, tfidf_auc, tfif_acc_per_class = tfidf_analyze(data_set_df, analyze_function, **vectorizer_args)

    return (count_acc, count_f1, count_auc, cout_acc_per_class), (tfidf_acc, tfidf_f1, tfidf_auc, tfif_acc_per_class)


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
        (count_acc, count_f1, count_auc, cout_acc_per_class), (tfidf_acc, tfidf_f1, tfidf_auc, tfif_acc_per_class) = evaluation(data_set_df, analyze_function, max_df=vocab_size, ngram_range=ngram_range, stop_words=stop_words)
        # print(tfif_acc_per_class)
        results.append({'Type': eval_type, 'Model': 'Count', 'Accuracy': count_acc, 'F1 Score': count_f1, 'AUC': count_auc, 'AUC Class 1 Jacque Chirac': cout_acc_per_class[1], 'AUC Class -1 François Mitterrand': tfif_acc_per_class[0]})
        results.append({'Type': eval_type, 'Model': 'TF-IDF', 'Accuracy': tfidf_acc, 'F1 Score': tfidf_f1, 'AUC': tfidf_auc, 'AUC Class 1 Jacque Chirac': tfif_acc_per_class[1], 'AUC Class -1 François Mitterrand': tfif_acc_per_class[0]})

    results_df = pd.DataFrame(results)
    return results_df

