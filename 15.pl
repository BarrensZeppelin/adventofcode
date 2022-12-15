:- use_module(library(clpfd)).
:- dynamic beacon/2, sensor/3.

read_input :-
	read_line_to_string(user_input, Line),
	(Line == end_of_file -> !;
		re_foldl([Match, [Num|Tail], Tail]>>(get_dict(0, Match, V), number_string(Num, V)),
			"-?\\d+", Line, [Sx, Sy, Bx, By], [], []),
		assertz(beacon(Bx, By)),
		Dist is abs(Bx - Sx) + abs(By - Sy),
		assertz(sensor(Sx, Sy, Dist)),
		read_input).

prdom(X) :-
	copy_term([X], [X], Gs),
	writeln(Gs).

ximp(Y, C) :-
	% Collect ranges of covered grid points on the line y=Y
	findall(L..R, (
		sensor(Sx, Sy, Dist), Rem is Dist - abs(Sy - Y),
		L is Sx-Rem, R is Sx+Rem, L =< R
	), [D1 | CTail]),
	% Union the ranges
	foldl([A,B,A\/B]>>true, CTail, D1, Dom),
	C in Dom.

main :-
	read_input, !,
	P1Y is 2000000,
	%P1Y is 10,
	ximp(P1Y, P1V),
	% Remove known beacons on the line
	bagof(true, X^(beacon(X, P1Y), P1V #\= X), _),
	fd_size(P1V, P1),
	format("Part 1: ~d~n", [P1]),
	MAX is 4000000,
	%MAX is 20,
	[X, Y] ins 0..MAX,
	% Downwards is a bit faster on my input
	% It should terminate within a minute or so
	labeling([down], [Y]),
	ximp(Y, XD), fd_dom(XD, Dom),
	#\ X in Dom, indomain(X), !,
	/*
	 * This code seems cleaner than the above, but unfortunately
	 * it doesn't seem to terminate. At least not within a couple
	 * of hours...
	bagof(true, Sx^Sy^Dist^(
		sensor(Sx, Sy, Dist),
		abs(X - Sx) + abs(Y - Sy) #> Dist), _), !,
	*/
	format("Part 2: ~d~n", [X*MAX + Y]).
