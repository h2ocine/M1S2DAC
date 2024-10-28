import subprocess
import parserpddl
import re
from tabulate import tabulate
import time
import sys


class ConsoleColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def generate_plan(path_domain, path_problem, n_max=50):

    # Path du planificateur 
    path_planificateur = "./planificateur_asp.lp"
    # Path du solveur
    path_solveur = "./solveur/clingo-4.4.0-x86_64-linux/clingo"

    # Iterations sur le nombre d'étapes
    n = 0
    while n < n_max: 
        file_path = './planificateur_asp.lp'
        new_line = f'#const n={n}. % Nombre d"etapes'
        modify_fifth_line(file_path, new_line)
        # Construction de la commande
        command = f"{path_solveur}  {path_domain} {path_problem} {path_planificateur}"

        # Exécution de la commande et récupération de la sortie
        try:
            output = subprocess.run(command, capture_output=True ,shell=True)
        except subprocess.TimeoutExpired:
            #print(f"Aucun plan trouvé pour n={n}")
            n += 1
            continue
        # Recuperation du résultat 
        return_code = output.returncode

        # Si le code return 10 ou 30 le code est satisfiable  alors on formate et print le résultat
        if return_code == 10 or return_code==30:
            print(f"{ConsoleColors.OKGREEN}Plan trouvé pour n={n}:{ConsoleColors.ENDC}\n")
            format_output(output, verbose = True)
            # Process the output or return it as needed
            break
        else:
            print(f"{ConsoleColors.FAIL}Pas de plan trouvé pour n={n}{ConsoleColors.ENDC}\n")

        n += 1


def format_output(output, verbose=False):
    # Récuperation du plan généré
    output_str = output.stdout.decode('utf-8')
    plan_start_index = output_str.index('perform(')
    plan_end_index = output_str.index('SATISFIABLE')
    plan_str = output_str[plan_start_index:plan_end_index].strip()
    plan_steps = plan_str.split(' ')
    plan_steps_transformed = []
    for word in plan_steps:
        # Utilisation de regex pour le formatage
        pattern = r"perform\((.*),(\d+)\)"
        match = re.match(pattern, word)
        if match:
            task = match.group(1)
            time = match.group(2)
            action = task.strip()
            step = time.strip()
            # Plan composé de l'action et du step correspondant
            plan_steps_transformed.append([action,int(step)])

    if verbose:
        print(f"\n{ConsoleColors.OKGREEN}Plan généré : {ConsoleColors.ENDC}\n")
        sorted_steps = sorted(plan_steps_transformed, key=lambda x: x[1])
        table = [[f"{ConsoleColors.BOLD}Action{ConsoleColors.ENDC}", f"{ConsoleColors.BOLD}Etape{ConsoleColors.ENDC}"]]  # Initialize the table with the header row

        # Iterate over each plan step and add a row to the table
        for action, time in sorted_steps:
            table.append([action, time])

        # Print the table using the tabulate function
        print(tabulate(table, headers="firstrow", tablefmt="simple_grid"))

    return plan_steps_transformed

def modify_fifth_line(file_path, new_line):

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # modification de la 5eme ligne ( qui correspond au const n dans le planneur asp )
    if len(lines) >= 5:
        lines[4] = new_line + '\n'  

        with open(file_path, 'w') as file:
            file.writelines(lines)
    else:
        print("The file does not have at least five lines.")


def main():

    """ 
    Donner en argument :

        - path du domain
        - path du problem                   
        - path de l'asp et du nom du monde exp : ./asp/NomMonde

    """
    if len(sys.argv) == 4:
        # Récupérez les arguments
        path_domain = sys.argv[1]
        path_problem = sys.argv[2]
        asp_file = sys.argv[3]
    else:
        print(" Pas d'argument donné, remplissage par defaut.\n\n")
        path_domain = "./pddl/blockWorld-domain.pddl"
        path_problem = "./pddl/blockWorld-problem.pddl"
        asp_file="./asp/blockWorld"
    
    

    # Création des fichiers ASP
    start_time_parser = time.time()
    parserpddl.create_asp_files(path_domain, path_problem, asp_file)
    parser_time = time.time() - start_time_parser

    # Génération du plan
    start_time_planner = time.time()
    generate_plan(f"{asp_file}_domain.lp", f"{asp_file}_problem.lp")
    planner_time = time.time() - start_time_planner

    total_time = parser_time + planner_time

    # Affichage des temps d'execution sous forme de tableau
    table = [
        ["Temps du parser", parser_time],
        ["Temps du planner", planner_time],
        ["Temps total", total_time]
    ]
    headers = [f"{ConsoleColors.BOLD}Étape{ConsoleColors.ENDC}", f"{ConsoleColors.BOLD}Temps (secondes){ConsoleColors.ENDC}"]
    print("\n\nTemps d'éxecution :\n")
    print(tabulate(table, headers, tablefmt="simple_grid"))

if __name__ == "__main__":
    main()











