:- use_module(library(clpfd)).

re_all(Pat, L, Ns) :- re_foldl([re_match{0:S}, [S|T], T]>>true, Pat, L, Ns, [], [capture_type(term)]).

main :-
    read_string(user_input, _, S), split_string(S, "\n", "\n", Lines),
    maplist(re_all("\\d+"), Lines, Input1),
    !,
    solve(Input1, Part1),
    format("Part 1: ~d\n", [Part1]),

    maplist(re_all("\\d"), Lines, Digits),
    maplist([Ds, [Squashed]]>>foldl([D, Acc, Res]>>(Res is Acc * 10 + D), Ds, 0, Squashed), Digits, Input2),
    solve(Input2, Part2),
    format("Part 2: ~d\n", [Part2]).

solve(T, D, W) :-
    Hi is T // 2,
    P in 0..Hi, P * (T - P) #> D,
    once(labeling([min(P)], [P])),
    W is T - 2 * P + 1.

solve([Times, Distances], Res) :-
    maplist(solve, Times, Distances, Wins),
    foldl([W, Acc, Res]>>(Res is Acc * W), Wins, 1, Res).
