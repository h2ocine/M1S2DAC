%%% planeS (DOMAIN) %%%
%%% DÃ©claration des prÃ©dicats (domain)
pred(at(X,Y)) :- plane(X),airport(Y).
pred(at(X,Y)) :- cargo(X),airport(Y).
pred(in(X,Y)) :- cargo(X),plane(Y).
pred(empty(X)) :- plane(X).

%%% DÃ©claration de l'action load
action(load(X,Y,Z)) :- cargo(X), plane(Y), airport(Z).

% PrÃ©conditions
pre(load(X,Y,Z),at(X,Z)) :- action(load(X,Y,Z)).
pre(load(X,Y,Z),at(Y,Z)) :- action(load(X,Y,Z)).
pre(load(X,Y,Z),empty(Y)) :- action(load(X,Y,Z)).

% Effets
del(load(X,Y,Z),at(X,Z)) :- action(load(X,Y,Z)).
del(load(X,Y,Z),empty(Y)) :- action(load(X,Y,Z)).
add(load(X,Y,Z),in(X,Y)) :- action(load(X,Y,Z)).

%%% DÃ©claration de l'action fly
action(fly(X,Y,Z)) :- plane(X), airport(Y), airport(Z).

% PrÃ©conditions
pre(fly(X,Y,Z),at(X,Y)) :- action(fly(X,Y,Z)).

% Effets
del(fly(X,Y,Z),at(X,Y)) :- action(fly(X,Y,Z)).
add(fly(X,Y,Z),at(X,Z)) :- action(fly(X,Y,Z)).

%%% DÃ©claration de l'action unload
action(unload(X,Y,Z)) :- cargo(X), plane(Y), airport(Z).

% PrÃ©conditions
pre(unload(X,Y,Z),in(X,Y)) :- action(unload(X,Y,Z)).
pre(unload(X,Y,Z),at(Y,Z)) :- action(unload(X,Y,Z)).

% Effets
del(unload(X,Y,Z),in(X,Y)) :- action(unload(X,Y,Z)).
add(unload(X,Y,Z),at(X,Z)) :- action(unload(X,Y,Z)).
add(unload(X,Y,Z),empty(Y)) :- action(unload(X,Y,Z)).



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% planeS (PROBLEM) %%%
%%% DÃ©claration des objets (problem)
#const n=12.
time(0..n).
plane(a1;a2).
cargo(c1;c2;c3).
airport(teg;cdg;bar).


%%% Etat initial (problem)
init(at(a1,teg)).
init(at(a2,teg)).
init(at(c1,teg)).
init(at(c2,teg)).
init(at(c3,teg)).
init(empty(a1)).
init(empty(a2)).

%%% But (problem)
but(at(a1,cdg)).
but(at(a2,cdg)).
but(at(c1,bar)).
but(at(c2,bar)).
but(at(c3,bar)).


% SOLUTION
% Models       : 5940
% Calls        : 1
% Time         : 0.891s (Solving: 0.88s 1st Model: 0.12s Unsat: 0.04s)
% CPU Time     : 0.880s

% perform(load(c1,a2,teg),0) 
% perform(load(c3,a1,teg),1)   
% perform(fly(a1,teg,bar),2)  
% perform(unload(c3,a1,bar),3)  
% perform(fly(a1,bar,teg),4) 
% perform(load(c2,a1,teg),5)  
% perform(fly(a1,teg,bar),6)  
% perform(unload(c2,a1,bar),7) 
% perform(fly(a2,teg,bar),8) 
% perform(unload(c1,a2,bar),9) 
% perform(fly(a2,bar,cdg),10) 
% perform(fly(a1,bar,cdg),11) 

0.0: (load c3 a2 teg)
1.0: (fly a2 teg bar)
2.0: (unload c3 a2 bar)
3.0: (fly a2 bar teg)
4.0: (load c1 a2 teg)
5.0: (fly a2 teg bar)
6.0: (load c2 a1 teg)
7.0: (fly a1 teg bar)
8.0: (unload c2 a1 bar)
9.0: (fly a1 bar cdg)
10.0: (unload c1 a2 bar)
11.0: (fly a2 bar cdg)

%unload(c3,a1,bar,2) unload(c1,a2,bar,5) unload(c2,a2,bar,10) load(c3,a1,teg,0) load(c1,a2,teg,3) load(c2,a2,teg,7) fly(a1,teg,bar,1) fly(a2,teg,bar,4) fly(a2,bar,teg,6) fly(a1,bar,cdg,8) fly(a2,teg,bar,9) fly(a2,bar,cdg,11)
%SATISFIABLE


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ASP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% Programmation des actions
%%% load
% prÃ©cond : il n'est pas permis de load un plane 
:-  load(X,Y,Z,T), not empty(Y,T). % si l'plane n'est pas vide
:-  load(X,Y,Z,T), not at(X,Z,T). % si la cargo nest pas Ã  l'aÃ©roport
:-  load(X,Y,Z,T), not at(Y,Z,T). % si l'plane nest pas Ã  l'aÃ©roport
  
% add : si on charge un plane, alors la cargo est in l'plane.
in(X,Y,T+1) :- load(X,Y,Z,T).

% del : si on charge un plane, alors un plane n'est plus vide et la cargo ne se at plus Ã  l'aÃ©roport.
% note : (ReprÃ©sentation des del, `-` reprÃ©sente la "vraie" nÃ©gation)
-empty(Y,T+1) :- load(X,Y,Z,T).
-at(X,Z,T+1) :- load(X,Y,Z,T).

%% fly
% precond : il  n'est pas permis de fly si l'plane ne se at pas Ã  l'aÃ©roport de dÃ©part
:- fly(X,Y,Z,T), not at(X,Y,T).

% add : si on vole d'un point Y Ã  Z, alors l'plane se at Ã  Z Ã  T+1.
at(X,Z,T+1) :- fly(X,Y,Z,T).

% del : si on vole d'un point Y Ã  Z, alors on ne se at plus Ã  Y Ã  T+1.
-at(X,Y,T+1) :- fly(X,Y,Z,T).

%% unload
% precond : il n'est pas permis de dÃ©load un plane si la cargo n'est pas in l'avuib et que l'avion n'est pas Ã  l'aÃ©roport.
:- unload(X,Y,Z,T), not in(X,Y,T).
:- unload(X,Y,Z,T), not at(Y,Z,T).

% add : si on dÃ©charge un plane, la cargo se at Ã  l'aÃ©roport d'arrivÃ©e et l'avion est vide.
at(X,Z,T+1) :- unload(X,Y,Z,T).
empty(Y,T+1) :- unload(X,Y,Z,T).

% del : si on dÃ©charge un plane, alors l'plane est vide, ie la cargo n'est plus in l'avion.
-in(X,Y,T+1) :- unload(X,Y,Z,T).

% Inertie :  si c'est vrai Ã  T et que ce n'est pas faux Ã  T+1, alors c'est vrai Ã  T+1
empty(X,T+1) :- empty(X,T), not -empty(X,T+1), T < n.

in(X,Y,T+1) :- in(X,Y,T), not -in(X,Y,T+1), T < n.

at(X,Y,T+1) :- at(X,Y,T), not -at(X,Y,T+1), T < n.

% contraintes d'integritÃ©
% :- at(X, Y, T), at(X, Z, T), plane(X), airport(Y), airport(Z), time(T), X != Y.
% :- in(X, Y, T), in(Z, Y, T), cargo(X), cargo(Z), plane(Y), time(T), X != Z.

% Si on veut faire une action a la fois pour chaque instant T
% { load(X,Y,Z,T) : cargo(X), plane(Y), airport(Z) } 1 :- time(T), T < n. 
% { fly(X,Y,Z,T) : plane(X), airport(Y), airport(Z) } 1 :- time(T), T < n.
% { unload(X,Y,Z,T) : cargo(X), plane(Y), airport(Z) } 1 :- time(T), T < n.
{ load(X, Y, Z, T) : cargo(X), plane(Y), airport(Z) ;
  fly(X, Y, Z, T) : plane(X), airport(Y), airport(Z) ; 
  unload(X, Y, Z, T) : cargo(X), plane(Y), airport(Z)} = 1 :- time(T), T < n.

%%% Etat initial
empty(a1,0).
empty(a2,0).
at(c1,teg,0).
at(c2,teg,0).
at(c3,teg,0).
at(a1,teg,0).
at(a2,teg,0).

%%% ReprÃ©sentation du but
:- not at(c1,bar,n).
:- not at(c2,bar,n).
:- not at(c3,bar,n).
:- not at(a1,cdg,n).
:- not at(a2,cdg,n).

#show load/4.
#show fly/4.
#show unload/4.