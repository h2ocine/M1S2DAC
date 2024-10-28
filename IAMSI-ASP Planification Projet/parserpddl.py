from pddl import parse_domain, parse_problem
import pddl.logic.predicates as pred
import pddl.logic.base as base

def create_asp_files(path_domain, path_problem, asp_file):

    # Lecture des fichier par la librairie pddl
    domain_str = parse_domain_str(path_domain)
    problem_str = parse_problem_str(path_problem)
    
    # Path des fichier pour l'ecriture
    domain_file_path = asp_file + "_domain.lp"
    problem_file_path = asp_file + "_problem.lp"
    
    with open(domain_file_path, "w") as domain_file:
        domain_file.write(domain_str)
    
    with open(problem_file_path, "w") as problem_file:
        problem_file.write(problem_str)



def parse_problem_str(path_problem):
    problem = parse_problem(path_problem)

    problem_str ="%Déclaration des objets (problem)\n\n"
    # Traitement des objects 
    for obj in problem.objects:
        problem_str += f'{str(sorted(obj.type_tags)[0])}({str(obj).lower()}).\n'
    
    problem_str += '\n'

    # Traitement des inits
    for inits in problem.init:
        if inits.arity > 0:
            problem_str += f'init({str(inits.name)}({", ".join(str(i.name).lower() for i in inits.terms)})).\n'
        else:
            problem_str += f'init({inits.name}{", ".join(str(i.name).lower() for i in inits.terms)}).\n'

    problem_str += "\n"

    # Traitement des buts
    for ope in problem.goal.operands:

        if ope.arity > 0:
            problem_str += f'but({str(ope.name)}({", ".join(str(o.name).lower() for o in ope.terms)})).\n'
        else:
            problem_str += f'but({ope.name}{", ".join(str(o.name).lower() for o in ope.terms)}).\n'
                
    return problem_str


def parse_domain_str(path_domain):

    # Declaration des prédicats : 
    domain = parse_domain(path_domain)
    domain_str = f"%{domain.name} :\n\n"
    predicate_str = ""

    # Traitement des  prédicats
    for predicate in domain.predicates:
        if len(predicate.terms)>0:
            predicate_str += f"pred({predicate.name}("
            predicate_str += ", ".join(str(param.name).upper() for param in predicate.terms)
            predicate_str += ")) "
            predicate_str += f':- ' 
            for i, term in enumerate(predicate.terms):
                predicate_str += f'{str(sorted(term.type_tags)[0])}({str(term.name).upper()})'
                if i < len(predicate.terms) - 1:
                    predicate_str += ', '
        else:
            predicate_str += f"pred({predicate.name})"
            
        predicate_str += ".\n"
    domain_str += predicate_str

    # Declaration des actions : 
    action_str =""
    for action in domain.actions:
        # Parametre : 

        action_str += f'action({action.name}({", ".join(str(act.name).upper() for act in action.terms)})'
        action_str += ') '
        action_str += f':- '
        for i, term in enumerate(action.terms):
            action_str += f'{str(sorted(term.type_tags)[0])}({str(term.name).upper()})'
            if i < len(action.terms) - 1:
                action_str += ', '

        action_str += ".\n"
        
        # Traitement des Preconditions : 
        precondition = action.precondition


        if isinstance(precondition,pred.Predicate):
            if precondition.arity > 0:
                action_str += f'pre({action.name}({", ".join(str(act.name).upper() for act in action.terms)}),{precondition.name}({", ".join(str(a.name).upper() for a in precondition.terms)}))'
            else:
                action_str += f'pre({action.name}({", ".join(str(act.name).upper() for act in action.terms)}),{precondition.name})'

            action_str += f' :- action({action.name}({", ".join(str(act.name).upper() for act in action.terms)})).'
            action_str += '\n'

        elif isinstance(precondition,base.And):
            ops = precondition.operands
            for operand in ops:
                if operand.arity > 0:
                    action_str += f'pre({action.name}({", ".join(str(act.name).upper() for act in action.terms)}),{operand.name}({", ".join(str(a.name).upper() for a in operand.terms)}))'
                else:
                    action_str += f'pre({action.name}({", ".join(str(act.name).upper() for act in action.terms)}),{operand.name})'

                action_str += f' :- action({action.name}({", ".join(str(act.name).upper() for act in action.terms)})).'
                action_str += '\n'

        # Traitement des Effects : 
        add_list = []
        del_list = []
        ops = action.effect.operands
        for ope in ops:
            if isinstance(ope,pred.Predicate):
                add_list.append(ope)
            elif isinstance(ope,base.Not):
                del_list.append(ope)
        
        # Add : 
        for add_operand in add_list:
            if add_operand.arity > 0:
                action_str += f'add({action.name}({", ".join(str(act.name).upper() for act in action.terms)}),{add_operand.name}({", ".join(str(a.name).upper() for a in add_operand.terms)}))'
            else:
                action_str += f'add({action.name}({", ".join(str(act.name).upper() for act in action.terms)}),{add_operand.name})'

            action_str += f' :- action({action.name}({", ".join(str(act.name).upper() for act in action.terms)})).'
            action_str += '\n'
        # Del
        for not_del_operand in del_list:
            del_operand = not_del_operand.argument
            if del_operand.arity > 0:
                action_str += f'del({action.name}({", ".join(str(act.name).upper() for act in action.terms)}),{del_operand.name}({", ".join(str(a.name).upper() for a in del_operand.terms)}))'
            else:
                action_str += f'del({action.name}({", ".join(str(act.name).upper() for act in action.terms)}),{del_operand.name})'
                
            action_str += f' :- action({action.name}({", ".join(str(act.name).upper() for act in action.terms)})).'
            action_str += '\n'
        action_str+='\n\n'
    domain_str += action_str

    return domain_str
