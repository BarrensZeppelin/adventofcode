:- use_module(library(clpfd)).

state(T, T, E, E) :- !.
state(T, TEnd, IElves, OElves) :-
	T < TEnd,

	assoc_to_keys(IElves, LElves),
	maplist(propose(T, IElves), LElves, Proposals),

	transpose_pairs(Proposals, TProp),
	group_pairs_by_key(TProp, PJoined),
	maplist([P-L, P-N]>>length(L, N), PJoined, PWCnt),
	list_to_assoc(PWCnt, NumProposals),
	maplist({NumProposals}/[E-P, O-0]>>(
		get_assoc(P, NumProposals, 1) -> O = P ; O = E
	), Proposals, UMElves),

	list_to_assoc(UMElves, MElves),

	NT is T + 1, !,
	state(NT, TEnd, MElves, OElves).

addp(X-Y, DX-DY, T, NX-NY) :- NX is X + DX * T, NY is Y + DY * T.
perp(X-Y, NX-X) :- NX is -Y.


dir(0, 0-(-1)).
dir(1, 0-1).
dir(2, (-1)-0).
dir(3, 1-0).
dir(X, V) :- X >= 4, T is X mod 4, dir(T, V).

any_nearby(Elves, EP) :-
	between(-1, 1, DX),
	between(-1, 1, DY),
	abs(DX) + abs(DY) =\= 0,
	addp(EP, DX-DY, 1, NP),
	get_assoc(NP, Elves, _), !.

occupied(Elves, P, D) :-
	between(-1, 1, T),
	addp(P, D, T, NP),
	get_assoc(NP, Elves, _).

propose(T, Elves, EP, EP-OP) :-
	any_nearby(Elves, EP),
	FT is T+3,
	between(T, FT, DT),
	dir(DT, D),
	addp(EP, D, 1, OP),
	perp(D, DP),
	%format("~w tries ~w~n", [EP, OP]),
	\+ occupied(Elves, OP, DP), !.

propose(_, _, EP, EP-EP).


read_input(Stream, Elves) :-
	read_string(Stream, _, Data),
	split_string(Data, "\n", "\n", Split),
	foldl([Line, Y-ILE, NY-OLE]>>(
		string_chars(Line, Chars),
		foldl({Y}/[C, X-IE, NX-OE]>>(
			(C == '.' -> OE = IE ;
				OE = [X-Y|IE]
			),
			NX is X + 1
		), Chars, 0-ILE, _-OLE),
		NY is Y + 1
	), Split, 0-[], _-UElves),
	maplist([P,P-0]>>true, UElves, PUElves),
	list_to_assoc(PUElves, Elves).

part2(Elves, T, OT) :-
	NT is T + 1,
	state(T, NT, Elves, NElves),
	(Elves == NElves -> OT = NT ;
		part2(NElves, NT, OT)).

main :-
	%leash(-all), trace,
	%open("23.in", read, Stream),
	read_input(user_input, IElves),
	%close(Stream),

	state(0, 10, IElves, OElves),
	aggregate_all(
		p(min(X), max(X), min(Y), max(Y), count),
		gen_assoc(X-Y, OElves, _),
		p(X1, X2, Y1, Y2, NumElves)
	),
	format("Part 1: ~d~n", [(X2 - X1 + 1) * (Y2 - Y1 + 1) - NumElves]),

	part2(OElves, 10, P2),
	format("Part 2: ~d~n", [P2]).
