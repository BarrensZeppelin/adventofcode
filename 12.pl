:- table dp(_,_,sum).

main :-
    read_string(user_input, _, S), split_string(S, "\n", "\n", Lines),
    maplist([Line, L-Cs]>>(
        split_string(Line, " ", "", [Left, CSS]), string_codes(Left, L),
        split_string(CSS, ",", "", CsL), maplist(number_string, Cs, CsL)
    ), Lines, Input),

    aggregate_all(sum(R), (member(L-Cs, Input), solve(L, Cs, R)), Part1),
    format("Part 1: ~d~n", [Part1]),

    aggregate_all(sum(R), (
        member(L-Cs, Input),
        append([L, [0'?], L, [0'?], L, [0'?], L, [0'?], L], NewL),
        append([Cs, Cs, Cs, Cs, Cs], NewCs),
        solve(NewL, NewCs, R)
    ), Part2),
    format("Part 2: ~d~n", [Part2]).

solve(L, Cs, R) :- append(L, [0'.], I), dp(I, Cs, R).

dp([], [], 1).
dp([], [_|_], 0).
dp([A|L], Cs, R) :- A \= 0'#, dp(L, Cs, R).
dp([A|L], [C|Cs], R) :- A \= 0'.,
    length(Head, C),
    append(Head, [CT|Tail], [A|L]),
    CT \= 0'#, \+ member(0'., Head),
    dp(Tail, Cs, R).
