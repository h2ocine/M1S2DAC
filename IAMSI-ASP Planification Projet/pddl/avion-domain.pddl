(define (domain planeS)
(:requirements :strips :typing)
(:types cargo plane airport - object)

(:predicates
  (at ?x - cargo ?y - airport)
  (at ?x - plane ?y - airport)
  (in ?x - cargo ?y - plane)
  (empty ?x - plane)
)

(:action load
  :parameters (?x - cargo ?y - plane ?z - airport)
  :precondition (and (at ?x ?z) (at ?y ?z) (empty ?y))
  :effect (and (not (at ?x ?z)) (not (empty ?y)) (in ?x ?y))
)

(:action fly
  :parameters (?x - plane ?y - airport ?z - airport)
  :precondition (at ?x ?y)
  :effect (and (not (at ?x ?y)) (at ?x ?z))
)

(:action unload
  :parameters (?x - cargo ?y - plane ?z - airport)
  :precondition (and (in ?x ?y) (at ?y ?z))
  :effect (and (not (in ?x ?y)) (at ?x ?z) (empty ?y))
)
)