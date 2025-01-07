:- use_module(library(clpfd)), use_module(library(prolog_stack)).
:- dynamic monkey/2.

read_input :-
	read_line_to_string(user_input, Line),
	(Line == end_of_file -> !;
		split_string(Line, ":", " ", [Name, OpS]),
		split_string(OpS, " ", "", Op),
		assertz(monkey(Name, Op)),
		read_input).

eval(Name, Res, Humn) :-
	(monkey(Name, Op) -> do_op(Op, Res, Humn);
	% Hack for part 2
	 Name == "humn" -> Res = Humn).

do_op([V], Res, Humn) :- (string(V) -> number_string(Res, V);
						  Res = V, Humn = V).
do_op([A, Op, B], Res, Humn) :-
	eval(A, VA, Humn), eval(B, VB, Humn),
	atom_string(F, Op),
	(F == / -> VB * Res #= VA ;
		Term =.. [F, VA, VB],
		Res #= Term).

main :-
	read_input,
	eval("root", P1, _),
	format("Part 1: ~d~n", [P1]),

	retract(monkey("humn", _)),
	monkey("root", [A, _, B]),
	eval(A, VA, Humn), eval(B, VB, Humn), VA #= VB,
	format("Part 2: ~d~n", [Humn]).
