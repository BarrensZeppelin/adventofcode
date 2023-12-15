:- use_module(library(clpfd)).

main :-
    read_string(user_input, _, S), split_string(S, "\n", "\n", Lines),
    maplist(string_codes, Lines, LineCodes),

    solve(LineCodes, 2, Part1),
    solve(LineCodes, 1000000, Part2),
    format("Part 1: ~d~nPart 2: ~d~n", [Part1, Part2]).

solve(L, Mul, R) :-
    one_way(L, Mul, 0, 0, R1),
    transpose(L, L2),
    one_way(L2, Mul, 0, 0, R2),
    R is R1 + R2.

one_way([], _, _, _, 0).
one_way([Line | Tail], Mul, Gal, Sum, R) :-
    aggregate_all(count, member(0'#, Line), Count),
    NGal is Gal + Count,
    (Count =:= 0 -> NSum is Sum + NGal * Mul; NSum is Sum + NGal),
    one_way(Tail, Mul, NGal, NSum, NR),
    R is NR + Count * Sum.

