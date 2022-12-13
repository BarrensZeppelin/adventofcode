:- use_module(library(clpfd)).
:- dynamic value/2.
:- table path(_, _, min) as subsumptive, can_traverse/2.

height('S', H) :- height(a, H).
height('E', H) :- height(z, H).
height(C, H) :- C \= 'E', C \= 'S',
	char_code(C, Code),
	H is Code - 97.

main :-
	% This program takes ~30s to execute, and I don't know enough about Prolog
	% to fix it. Perhaps it would help to pre-constrain the points that are
	% passed to can_traverse with between.
	read_string(user_input, _, Data),
	split_string(Data, "\n", "\n", Lines),
	length(Lines, H),
	forall(between(1, H, Y),
		(
			nth1(Y, Lines, Line), string_chars(Line, Chars), length(Chars, W),
			forall(between(1, W, X),
				(
					nth1(X, Chars, C),
					assertz(value(X-Y, C))
				)
			)
		)
	), !,
	%leash(-all), trace,
	value(Start, 'S'),
	value(End, 'E'),
	path(Start, End, P1),
	format("Part 1: ~w~n", [P1]),
	aggregate_all(min(C), (value(St, a), path(St, End, C)), P2),
	format("Part 2: ~w~n", [P2]).

path(P, P, 0).
% This rule is not good!
%path(A, B, X) :- path(B, A, X).
path(A, C, X) :-
	can_traverse(A, B),
	path(B, C, V1),
	X is V1 + 1.

can_traverse(A, B) :-
	X1-Y1 = A, X2-Y2 = B,
	DX #= X2 - X1, DY #= Y2 - Y1,
	abs(DX) + abs(DY) #= 1,
	H2 - H1 #=< 1,
	value(A, V1), height(V1, H1),
	value(B, V2), height(V2, H2).
