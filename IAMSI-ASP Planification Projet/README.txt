<h1>Description du projet<h1>

Ce projet consiste en la résolution de problèmes de planification en utilisant des fichiers PDDL, des solveurs ASP (Answer Set Programming) et un programme principal en Python.

Le répertoire contient les fichiers suivants :

solveur/ : Répertoire contenant le solveur ASP clingo.

asp_manuel/ : Fichiers ASP manuellement créés pour résoudre les problèmes de planification.

asp/ : Répertoire contenant les fichiers ASP générés à partir du programme principal.

parserpddl.py : Fichier Python contenant la fonction de conversion de PDDL en ASP.

main.py : Fichier Python contenant le programme principal qui exécute le processus de transformation des fichiers PDDL en parametres en fichiers ASP, puis appelle le solveur ASP pour résoudre les problèmes de planification.



<h1>Instructions d'utilisation<h1>

Assurez-vous que le fichier clingo dans le répertoire solveur/clingo-4.4.0-x86_64-linux/ a les permissions d'exécution. Si ce n'est pas le cas, exécutez la commande suivante dans le terminal pour donner les permissions d'exécution : chmod 777 ./solveur/clingo-4.4.0-x86_64-linux/clingo.

Pour résoudre le problème de planification dans le monde des blocs, exécutez la commande suivante dans le terminal :
python3 main.py ./plans/blockWorld-domain.pddl ./plans/blockWorld-problem.pddl ./asp/blockWorld


Pour résoudre le problème de planification dans le domaine des avions, exécutez la commande suivante dans le terminal :
python3 main.py ./plans/avion-domain.pddl ./plans/avion-problem.pddl ./asp/avion

Pour exécuter un autre fichier PDDL, utilisez la commande suivante dans le terminal : 
python3 main.py [nom_fichier_domain.pddl] [nom_fichier_problem.pddl] [path_fichier_asp_généré]

Assurez-vous d'avoir les dépendances nécessaires installées pour exécuter le programme principal.

