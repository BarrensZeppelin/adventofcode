read_lines(L) :-
	read_line_to_string(user_input, Line),
	(Line == end_of_file -> !, L = [] ;
		read_lines(Tail),
		(Line == "" -> L = Tail ;
			term_string(Term, Line),
			is_list(Term),
			L = [Term | Tail]
		)
	).


cmp([], [], '=').
cmp([], [_|_], '<').
cmp([_|_], [], '>').
cmp([A|As], [B|Bs], C) :-
	cmp(A, B, T),
	(T == '=' ->
		cmp(As, Bs, C) ;
		C = T).

cmp(X, Y, C) :- number(X), number(Y), compare(C, X, Y).
cmp(X, Y, C) :- number(X), is_list(Y), cmp([X], Y, C).
cmp(X, Y, C) :- is_list(X), number(Y), cmp(X, [Y], C).


main :-
	read_lines(Lines),
	%leash(-all), trace,
	aggregate_all(sum(I+1),
		(append([Prefix, [A, B], _], Lines), length(Prefix, P),
			divmod(P, 2, I, 0), cmp(A, B, '<')),
	 	P1),
	format("Part 1: ~d~n", [P1]),

	aggregate_all(count, (member(A, Lines), cmp(A, 2, '<')), X),
	aggregate_all(count, (member(A, [2|Lines]), cmp(A, 6, '<')), Y),
	format("Part 2: ~d~n", [(X+1) * (Y+1)]).
