:- use_module(library(clpfd)).

main :-
    read_string(user_input, _, Data),
    split_string(Data, "\n", "\n", L),
    maplist(part1, L, LP1),
    sum_list(LP1, P1),
    writeln(P1),
    part2(L, 0).

prio(C, X) :-
    (C #>= 96 -> X is C-96) ; X is (C-64)+26.

part1(S, Z) :-
    string_length(S, L),
    HL is L // 2,
    sub_string(S, 0, HL, _, S1),
    sub_string(S, HL, HL, _, S2),
    string_code(_, S1, X),
    string_code(_, S2, X),
    prio(X, Z).


common([], _).
common([H | T], X) :-
    string_code(_, H, X),
    common(T, X).

part2([], X) :- writeln(X).
part2([A, B, C | T], X) :-
    common([A, B, C], D),
    prio(D, Y),
    Z is X + Y,
    part2(T, Z).
