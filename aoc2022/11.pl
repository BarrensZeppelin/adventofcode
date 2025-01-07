:- use_module(library(prolog_stack)).

odds([X], [X]).
odds([X, _ | Tail], [X | NTail]) :- odds(Tail, NTail).

ints(String, Ints) :-
	string(String),
	re_foldl([re_match{0:I}, [I | T], T]>>true, "\\d+"/t, String, Ints, [], []).

parse_monkey(
	Spec,
	monkey{id: Id, items: Items, test: Div, tId: TId, fId: FId, op: Op, throws: 0}
) :-
	split_string(Spec, "\n", "\n", [IDLine, ItemLine, OpLine, TestLine, TLine, FLine]),
	ints(IDLine, [Id]),
	ints(ItemLine, Items),
	re_matchsub("new = (\\S+) (\\S) (\\S+)"/t, OpLine, Sub, []),
	bagof(X, I^(between(1, 3, I), get_dict(I, Sub, X)), Op),
	ints(TestLine, [Div]),
	ints(TLine, [TId]),
	ints(FLine, [FId]).


getv(Old, old, Old) :- !.
getv(_, X, X) :- number(X).

finalize(p1, V, New) :- New is V // 3.
finalize(p2(Mod), V, New) :- New is V mod Mod.

eval(Mode, Monkey, Old, New) :-
	[A, Op, B] = Monkey.op,
	maplist(getv(Old), [A, B], [AV, BV]),
	Term =.. [Op, AV, BV],
	finalize(Mode, Term, New).

do_throw(Mode, Monkey, V, A, B) :-
	eval(Mode, Monkey, V, NV),
	(NV mod Monkey.test =:= 0 -> NId = Monkey.tId ; NId = Monkey.fId),
	nth0(NId, A, PM, Rest),
	NM = PM.put(items, [NV | PM.items]),
	nth0(NId, B, NM, Rest).


process_monkey(Mode, Id, IState, OState) :-
	nth0(Id, IState, Monkey),

	foldl(do_throw(Mode, Monkey), Monkey.items, IState, TState),
	length(Monkey.items, N),

	nth0(Id, TState, Monkey, TStateWOMonkey),
	NThrows is Monkey.throws + N,
	NMonkey = Monkey.put([throws:NThrows, items:[]]),
	nth0(Id, OState, NMonkey, TStateWOMonkey).


rounds(_, 0, A, A).
rounds(Mode, T, A, C) :- T > 0,
	NT is T - 1,
	maplist(get_dict(id), A, Ids),
	foldl(process_monkey(Mode), Ids, A, B),
	rounds(Mode, NT, B, C).

extract_ans(Monkeys, Ans) :-
	maplist(get_dict(throws), Monkeys, Throws),
	sort(0, @>=, Throws, [A, B | _]),
	Ans is A * B.


main :-
	read_string(user_input, _, Data),
	re_split("\n\n", Data, Split),
	odds(Split, MonkeySpecs),
	maplist(parse_monkey, MonkeySpecs, Monkeys),

	rounds(p1, 20, Monkeys, P1Monkeys),

	extract_ans(P1Monkeys, P1),
	format("Part 1: ~d~n", [P1]),

	maplist(get_dict(test), Monkeys, Tests),
	foldl([T, O, R]>>(R is O * T), Tests, 1, Mod),
	rounds(p2(Mod), 10_000, Monkeys, P2Monkeys),

	extract_ans(P2Monkeys, P2),
	format("Part 2: ~d~n", [P2]).
