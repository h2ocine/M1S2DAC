% MONDE DES BLOCS 

%%%%%%%%%%
% DOMAIN %
%%%%%%%%%%

% DÃ©claration des prÃ©dicats (domain) 
pred(on(X,Y)):-block(X;Y).
pred(ontable(X)):-block(X).
pred(clear(X)):-block(X).
pred(handempty).
pred(holding(X)):-block(X).

% DÃ©claration des actions 

% Pickup 
action(pickup(X)):-block(X).
% Preconditions 
pre(pickup(X),clear(X)):-action(pickup(X)).
pre(pickup(X),ontable(X)):-action(pickup(X)).
pre(pickup(X),handempty):-action(pickup(X)).
% Effects 
del(pickup(X),clear(X)):-action(pickup(X)).
del(pickup(X),ontable(X)):-action(pickup(X)).
del(pickup(X),handempty):-action(pickup(X)).
add(pickup(X),holding(X)):-action(pickup(X)).


% Putdown 
action(putdown(X)):-block(X).
% Preconditions 
pre(putdown(X),holding(X)):-action(putdown(X)).
% Effects 
add(putdown(X),clear(X)):-action(putdown(X)).
add(putdown(X),ontable(X)):-action(putdown(X)).
add(putdown(X),handempty):-action(putdown(X)).
del(putdown(X),holding(X)):-action(putdown(X)).


% Stack 
action(stack(X,Y)):-block(X;Y).
% Preconditions 
pre(stack(X,Y), holding(X)):-action(stack(X,Y)).
pre(stack(X,Y), clear(Y)):-action(stack(X,Y)).
% Effects 
add(stack(X,Y), on(X,Y)):-action(stack(X,Y)).
add(stack(X,Y), clear(X)):-action(stack(X,Y)).
add(stack(X,Y), handempty):-action(stack(X,Y)).
del(stack(X,Y), holding(X)):-action(stack(X,Y)).
del(stack(X,Y), clear(Y)):-action(stack(X,Y)).


% Unstack 
action(unstack(X,Y)):-block(X;Y).
% Preconditions 
pre(unstack(X,Y), on(X,Y)):-action(unstack(X,Y)).
pre(unstack(X,Y), clear(X)):-action(unstack(X,Y)).
pre(unstack(X,Y), handempty):-action(unstack(X,Y)).
% Effects 
add(unstack(X,Y), holding(X)):-action(unstack(X,Y)).
add(unstack(X,Y), clear(Y)):-action(unstack(X,Y)).
del(unstack(X,Y), on(X,Y)):-action(unstack(X,Y)).
del(unstack(X,Y), clear(X)):-action(unstack(X,Y)).
del(unstack(X,Y), handempty):-action(unstack(X,Y)).


%%%%%%%%%%%
% PROBLEM %
%%%%%%%%%%%

% DÃ©claration des objets (problem) ou constantes (domain) 
block(a;b;c;d).

% Etat initial 
init(clear(b);clear(c)).
init(ontable(a);ontable(d)).
init(on(c,d);on(b,a)).
init(handempty).

% But 
but(ontable(d)).
but(on(a,b)).
but(on(b,c)).
but(on(c,d)).
but(clear(a)).
but(handempty).


