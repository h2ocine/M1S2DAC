%Déclaration des objets (problem)

block(d).
block(c).
block(a).
block(b).

init(ontable(a)).
init(clear(b)).
init(handempty).
init(ontable(d)).
init(on(c, d)).
init(on(b, a)).
init(clear(c)).

but(clear(a)).
but(on(a, b)).
but(on(b, c)).
but(on(c, d)).
but(handempty).
but(ontable(d)).
