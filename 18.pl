:- use_module(library(clpfd)).
:- dynamic lava/1.
:- table air(_, _).

read_input(Stream) :-
	read_line_to_string(Stream, Line),
	(Line == end_of_file -> !;
		split_string(Line, ",", "", Split),
		maplist(number_string, [X,Y,Z], Split),
		assertz(lava(p(X, Y, Z))),
		read_input(Stream)).

neighbours(p(X, Y, Z), p(A, B, C)) :-
	abs(A-X) + abs(B-Y) + abs(C-Z) #= 1,
	label([A, B, C]).

inbounds(p(X, Y, Z), b(XB, YB, ZB)) :- X in XB, Y in YB, Z in ZB.


air(P, Bounds) :- \+ lava(P),
	(\+ inbounds(P, Bounds) ; (neighbours(P, NP), air(NP, Bounds))).

main :-
	read_input(user_input),
	aggregate_all(count, (lava(P), neighbours(P, NP), \+ lava(NP)), P1),
	format("Part 1: ~d~n", [P1]),
	aggregate_all(
		b(min(X), min(Y), min(Z), max(X), max(Y), max(Z)),
		lava(p(X, Y, Z)),
		b(X1, Y1, Z1, X2, Y2, Z2)
	),
	Bounds = b(X1..X2, Y1..Y2, Z1..Z2),
	aggregate_all(count, (lava(P), neighbours(P, NP), air(NP, Bounds)), P2),
	format("Part 2: ~d~n", [P2]).

