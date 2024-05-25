Résumé du Code pour lineaire_1.py
Le fichier lineaire_1.py contient l'implémentation de la classe Linear, qui hérite de la classe abstraite Module. Cette classe représente une couche linéaire dans un réseau de neurones et inclut les fonctionnalités suivantes :

Initialisation des paramètres : avec différentes méthodes telles que zero, one, random, uniform, xavier, et lecun.
Passe avant (forward) : calcule les sorties en utilisant les poids et les biais.
Mise à jour des gradients : met à jour les gradients des poids et des biais en utilisant les deltas.
Rétropropagation des deltas : calcule le delta à rétropropager à la couche précédente.


Résumé du Code pour nonlineaire_2.py
Le fichier nonlineaire_2.py implémente plusieurs fonctions d'activation couramment utilisées dans les réseaux de neurones. Ces modules incluent :

TanH : Hyperbolic Tangent
Sigmoid : Sigmoïde
ReLU : Rectified Linear Unit
LeakyReLU : Leaky Rectified Linear Unit
ELU : Exponential Linear Unit
Softmax : Fonction Softmax pour les sorties de réseaux multiclasses
Chaque classe hérite de Activation, qui est une sous-classe de Module. Ces classes implémentent les méthodes forward et _backward pour calculer les sorties et les gradients respectivement.


Résumé du Code pour encapsulage_3.py
Le fichier encapsulage_3.py implémente la classe Sequential, qui permet d'encapsuler plusieurs modules en série. Cette classe hérite de Module et inclut les fonctionnalités suivantes :

Initialisation : prend une liste de modules à chaîner.
Ajout de modules : permet d'ajouter dynamiquement des modules à la séquence.
Passe avant (forward) : enchaîne les passes avant de chaque module.
Rétropropagation des gradients : enchaîne les mises à jour des gradients de chaque module.
Rétropropagation des deltas : enchaîne les calculs des deltas pour chaque module.
Mise à jour des paramètres : met à jour les paramètres de chaque module en fonction du gradient.
Gestion des paramètres : permet de définir et obtenir les paramètres des modules encapsulés.
Cette classe est essentielle pour construire des réseaux de neurones plus complexes en chaînant simplement plusieurs modules ensemble.


Résumé du Code pour multiclasse_4.py
Le fichier multiclasse_4.py introduit la gestion des problèmes de classification multiclasse en utilisant une couche avec activation Softmax. Les principales fonctionnalités incluent :

SoftmaxLayer : une classe encapsulant la fonction d'activation Softmax.
Utilisation d'une fonction de coût adaptée : comme l'entropie croisée (CELoss ou CCELoss).
Un exemple d'utilisation est également fourni, montrant comment construire un réseau de neurones pour un problème de classification multiclasse, effectuer une passe avant pour calculer les sorties et le coût, et effectuer une rétropropagation pour mettre à jour les gradients et les paramètres.


Résumé du Code pour compresse_5.py
Le fichier compresse_5.py implémente un autoencodeur, un type de réseau de neurones utilisé pour la réduction de dimension et la reconstruction de données. Les principales fonctionnalités incluent :

AutoEncoder : une classe qui hérite de Sequential et encapsule un encodeur et un décodeur.
Méthode parameters_sharing : permet de partager les paramètres entre l'encodeur et le décodeur en utilisant la transposée des poids.
Exemple d'utilisation : montre comment créer un autoencodeur, effectuer une passe avant pour encoder et décoder les données, et effectuer une rétropropagation pour mettre à jour les gradients et les paramètres.


Résumé du Code pour utils.py
Le fichier utils.py contient des fonctions utilitaires utilisées dans les différentes parties du projet :

onehot_encoding : encode des étiquettes sous forme de vecteurs one-hot.
softmax : calcule la fonction softmax pour normaliser les sorties des réseaux de neurones.


Résumé du Code pour module.py
Le fichier module.py contient la classe de base Module pour tous les modules de réseau de neurones. Les principales fonctionnalités incluent :

Initialisation des paramètres et des gradients
Méthode forward : à implémenter dans les sous-classes pour la passe avant.
Méthode backward_update_gradient : à implémenter dans les sous-classes pour mettre à jour le gradient.
Méthode backward_delta : à implémenter dans les sous-classes pour calculer le delta à rétropropager.
Sauvegarde des données d'entrée : pour une utilisation lors de la rétropropagation.
Mise à jour des paramètres : en utilisant le gradient calculé.
Réinitialisation des gradients : à zéro.

