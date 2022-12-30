:- dynamic map/3.

read_input(Stream, Y) :-
	read_line_to_string(Stream, Line),
	(Line == end_of_file -> compile_predicates([map/3]) ;
		string_chars(Line, Chars),
		length(Chars, W),
		W1 is W-1,
		numlist(0, W1, XS),
		maplist({Y}/[C, X]>>assertz(map(X, Y, C)), Chars, XS),
		NY is Y + 1,
		read_input(Stream, NY)
	).

dir('>', 1-0).
dir('<', (-1)-0).
dir('v', 0-1).
dir('^', 0-(-1)).
dir(' ', 0-0).

addp(X-Y, DX-DY, T, NX-NY) :-
	NX is X + DX * T, NY is Y + DY * T.

blizz(T, P, W, H) :-
	_-Y = P, 0 < Y, Y < H-1,
	dir(C, D), D \= ' ',
	addp(P, D, -T, NX-NY),
	CX is (NX - 1) mod (W - 2) + 1,
	CY is (NY - 1) mod (H - 2) + 1,
	map(CX, CY, C).


path(StartT, EndT, Start, End, W, H) :-
	bfs(StartT, EndT, [Start], End, W, H).

bfs(T, T, Frontier, End, _, _) :- member(End, Frontier), !.
bfs(StartT, EndT, Frontier, End, W, H) :-
	!, Frontier \= [],
	NT is StartT+1,
	setof(NP, neighbours(Frontier, NT, W, H, NP), NFrontier),
	bfs(NT, EndT, NFrontier, End, W, H).

inbounds(X-Y) :- map(X, Y, C), C \= '#'.

neighbours(Frontier, NT, W, H, NP) :-
	member(P, Frontier),
	dir(_, DP),
	addp(P, DP, 1, NP),
	inbounds(NP),
	\+ blizz(NT, NP, W, H).


main :-
	%open("24.in", read, Stream),
	read_input(user_input, 0),
	%close(Stream),

	aggregate(p(max(X+1), max(Y+1)), map(X, Y, _), p(W, H)),
	H1 is H-1,
	map(SX, 0, '.'),
	map(EX, H1, '.'),

	path(0, P1, SX-0, EX-H1, W, H),
	format("Part 1: ~d~n", [P1]),

	path(P1, ToStart, EX-H1, SX-0, W, H),
	path(ToStart, P2, SX-0, EX-H1, W, H),
	format("Part 2: ~d~n", [P2]).
