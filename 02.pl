main :-
    read_input(L),
    part1(L, 0),
    part2(L, L2),
    part1(L2, 0).

outcome(X, X, 3).
outcome(X, Y, S) :-
    (mod(X+1, 3) =:= Y -> S is 6) ; S is 0.

score(X, Y, S) :-
    outcome(X, Y, Z),
    S is Z + Y + 1.

part1([], X) :- writeln(X).
part1([[A, B] | Tail], X) :-
    score(A, B, Y),
    Z is X + Y,
    part1(Tail, Z).

part2([], _).
part2([[A, B] | Tail], [[A, M] | NTail]) :-
    M is mod(A + (B-1), 3),
    part2(Tail, NTail).

read_input(L) :-
    read_line_to_string(user_input, Line),
    (Line == end_of_file, L = [];
        split_string(Line, " ", "", [A, B]),
        char_code(A, C1),
        char_code(B, C2),
        I1 is C1 - 65,
        I2 is C2 - 88,
        read_input(Tail),
        L = [[I1, I2] | Tail]
    ).