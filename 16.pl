:- dynamic tunnel/2, rate/2.
:- set_prolog_flag(table_space, 4294967296).  % 4 GiB
:- table pressure(_, _, _, max), dist(_, _, min).

read_input(Stream) :-
	read_line_to_string(Stream, Line),
	(Line == end_of_file -> !;
		re_matchsub("Valve (?<name>\\w{2}) has flow rate=(?<rate>\\d+)", Line,
			re_match{0:_, name: Name, rate: RateS}, []),
		number_string(Rate, RateS),
		assertz(rate(Name, Rate)),
		re_split("valves? ", Line, [_, _, TunnelSpec]),
		re_foldl({Name}/[re_match{0:Next}, _, _]>>assertz(tunnel(Name, Next)),
			"\\w{2}", TunnelSpec, _, _, []),
		read_input(Stream)).

dist(A, A, 0).
dist(A, C, D) :- tunnel(A, B), dist(B, C, X), D is X+1.

pressure(Time, Set, P) :- pressure(Time, "AA", Set, P).

pressure(Time, _, [], 0) :- (Time >= 0 -> !).
pressure(Time, Pos, Rem, Pres) :- Time >= 0,
	select(NPos, Rem, NRem),
	dist(Pos, NPos, Dist),
	NTime is Time - Dist - 1,
	pressure(NTime, NPos, NRem, NScore),
	rate(NPos, Rate),
	Pres is NScore + Rate * NTime.


powerset([], []).
powerset([_|T], P) :- powerset(T,P).
powerset([H|T], [H|P]) :- powerset(T,P).

main :-
	% The program takes a couple of minutes to run.
	% I'm not sure how to improve it. 😕
	%open("16.in", read, Stream),
	read_input(user_input),
	%close(Stream),
	setof(Name, Rate^(rate(Name, Rate), Rate > 0), NonZ),
	aggregate_all(
		max(Score),
		(powerset(NonZ, Open), is_ordset(Open), pressure(30, Open, Score)),
		P1
	),
	format("Part 1: ~d~n", [P1]),
	aggregate_all(
		max(Score),
		(powerset(NonZ, Open), powerset(Open, Open1),
		 pressure(26, Open1, S1),
		 ord_subtract(Open, Open1, Open2),
		 pressure(26, Open2, S2),
		 Score is S1 + S2),
		P2
	),
	format("Part 2: ~d~n", [P2]).

