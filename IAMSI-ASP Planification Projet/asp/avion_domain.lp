%planeS :

pred(in(X, Y)) :- cargo(X), plane(Y).
pred(empty(X)) :- plane(X).
pred(at(X, Y)) :- cargo(X), airport(Y).
pred(at(X, Y)) :- plane(X), airport(Y).
action(unload(X, Y, Z)) :- cargo(X), plane(Y), airport(Z).
pre(unload(X, Y, Z),in(X, Y)) :- action(unload(X, Y, Z)).
pre(unload(X, Y, Z),at(Y, Z)) :- action(unload(X, Y, Z)).
add(unload(X, Y, Z),at(X, Z)) :- action(unload(X, Y, Z)).
add(unload(X, Y, Z),empty(Y)) :- action(unload(X, Y, Z)).
del(unload(X, Y, Z),in(X, Y)) :- action(unload(X, Y, Z)).


action(load(X, Y, Z)) :- cargo(X), plane(Y), airport(Z).
pre(load(X, Y, Z),at(X, Z)) :- action(load(X, Y, Z)).
pre(load(X, Y, Z),at(Y, Z)) :- action(load(X, Y, Z)).
pre(load(X, Y, Z),empty(Y)) :- action(load(X, Y, Z)).
add(load(X, Y, Z),in(X, Y)) :- action(load(X, Y, Z)).
del(load(X, Y, Z),at(X, Z)) :- action(load(X, Y, Z)).
del(load(X, Y, Z),empty(Y)) :- action(load(X, Y, Z)).


action(fly(X, Y, Z)) :- plane(X), airport(Y), airport(Z).
pre(fly(X, Y, Z),at(X, Y)) :- action(fly(X, Y, Z)).
add(fly(X, Y, Z),at(X, Z)) :- action(fly(X, Y, Z)).
del(fly(X, Y, Z),at(X, Y)) :- action(fly(X, Y, Z)).


