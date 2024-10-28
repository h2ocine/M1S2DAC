(define (problem planeS-problem)
  (:domain planeS)
  (:objects
    a1 a2 - plane
    c1 c2 c3 - cargo
    teg cdg bar - airport
  )

  (:init
    (at c1 teg)
    (at c2 teg)
    (at c3 teg)
    (at a1 teg)
    (at a2 teg)
    (empty a1)
    (empty a2)
  )

  (:goal
    (and
      (at a1 cdg)
      (at a2 cdg)
      (at c1 bar)
      (at c2 bar)
      (at c3 bar)
    )
  )
)