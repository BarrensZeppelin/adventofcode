:- dynamic value/2.

value(1, 1).

main :-
	read_input(1),
	aggregate_all(sum(T * X), (value(T, X), T mod 40 =:= 20), P1),
	format("Part 1: ~d~n", [P1]),
	%leash(-all), trace,
	forall(between(0, 5, Y),
		(forall(between(0, 39, X),
				(T is Y * 40 + X + 1, value(T, V),
				(abs(V - X) =< 1 -> C = "#" ; C = " "),
				write(C))),
			writeln("")
		)).

read_input(T) :-
    read_line_to_string(user_input, Line),
    ( Line == end_of_file -> !;
		split_string(Line, " ", "", Split),
		process(Split, T, NT),
		read_input(NT)
	).

process(["noop"], T, NT) :-
	value(T, PV),
	NT is T + 1,
	assertz(value(NT, PV)).

process(["addx", S], T, NT) :-
	process(["noop"], T, T1),
	value(T1, PV),
	NT is T1 + 1,
	number_string(N, S),
	NV is PV + N,
	assertz(value(NT, NV)).
