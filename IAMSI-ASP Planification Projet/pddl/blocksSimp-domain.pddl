(define (domain blockWorld)

    (:requirements :strips :typing :equality :conditional-effects :disjunctive-preconditions)
    ;;; conditional-effects : when
    ;;; disjonctive-preconditions : or
    (:types block - object)
    (:constants table - object)
    (:predicates 
        (on ?x - block ?y - object) ;;; on pose le bock x sur y 
        (clear ?x - object) ;;; on a de lespace sur x (x peut Ãªtre la table ou un block)
    )
    (:action  move
        :parameters (?x - block ?s ?d - object)
        :precondition (and (clear ?x) (on ?x ?s) (not (= ?s ?d)) (or (= ?d table) (clear ?d)))
        :effect (and
            (on ?x ?d) ;;; on met x sur d
            (when (not (= ?s table)) (clear ?s)) ;;; si s nest pas la table on clear s pour laisser de la place sur le block
            (not (on ?x ?s)) ;;; x nest plus sur s
            (when (not (= ?d table)) (not (clear ?d))) ;;; si on met x sur la table on ne la clear pas 
        )
    )
)

