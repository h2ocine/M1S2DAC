(define (problem blockPyramidProblem)
  (:domain blockWorld)
  (:objects A B C D E F - block)
  (:init
    (clear F)
    (ontable A)
    (on F E)
    (on E D)
    (on D C)
    (on C B)
    (on B A)
    (handempty)
  )
  (:goal
    (and
    (clear A)
    (clear D)
    (clear E)
    
    (ontable C)
    (ontable B)
    (ontable F)
    
    (on A C)
    (on E F)
    (on D B)

    (handempty)
    )
  )
)
