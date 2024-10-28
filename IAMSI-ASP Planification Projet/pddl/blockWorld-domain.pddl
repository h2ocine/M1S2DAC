(define (domain blockWorld)

    (:requirements :strips :typing)
    (:types block)
    (:predicates
    (on ?x - block ?y - block)
    (ontable ?x - block)
    (clear ?x - block)
    (handempty)
    (holding ?x - block)
    )

    (:action pickup
    ;;; action qui ramasse un bloc pose sur la table
    :parameters (?x - block)
    :precondition (and (clear ?x) (ontable ?x) (handempty))
    :effect (and (not (ontable ?x))
    (not (clear ?x))
    (not (handempty))
    (holding ?x))
    )

    (:action putdown
    ;;; action qui pose un bloc en main  sur la table
    :parameters (?x - block)
    :precondition (holding ?x)
    :effect (and (clear ?x) (ontable ?x) (handempty) 
    (not (holding ?x)))
    )

    (:action stack
    ;;; action qui pose un bloc en main  sur un bloc en sommet 
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (clear ?x) (on ?x ?y) (handempty) 
    (not (holding ?x)) (not (clear ?y)))
    )

    (:action unstack
    ;;; action qui ramasse un bloc poser sur un autre 
    :parameters (?x - block ?y - block)
    :precondition (and (clear ?x) (on ?x ?y) (handempty))
    :effect (and (holding ?x) (clear ?y)
    (not (on ?x ?y)) (not (clear ?x)) (not (handempty)))
    )
)
