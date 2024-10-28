%Déclaration des objets (problem)

airport(teg).
cargo(c3).
plane(a2).
airport(bar).
plane(a1).
cargo(c1).
airport(cdg).
cargo(c2).

init(at(c3, teg)).
init(empty(a1)).
init(at(a2, teg)).
init(at(a1, teg)).
init(empty(a2)).
init(at(c2, teg)).
init(at(c1, teg)).

but(at(a1, cdg)).
but(at(a2, cdg)).
but(at(c1, bar)).
but(at(c2, bar)).
but(at(c3, bar)).
