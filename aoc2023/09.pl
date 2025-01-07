main :-
    read_string(user_input, _, S), split_string(S, "\n", "\n", Lines),
    maplist([Line, Nums]>>(split_string(Line, " ", "", Split), maplist(number_string, Nums, Split)), Lines, Input),

    aggregate_all(r(sum(Next), sum(Prev)), (
        member(Nums, Input), solve(Nums, Next), reverse(Nums, Rev), solve(Rev, Prev)
    ), r(Part1, Part2)),
    format("Part 1: ~d~nPart 2: ~d~n", [Part1, Part2]).

differences([A, B | Tail], [Diff | Diffs]) :- Diff is B - A, differences([B | Tail], Diffs).
differences([_], []).

solve([0], 0).
solve(Nums, Next) :-
    differences(Nums, Diffs), last(Nums, Last),
    solve(Diffs, N2), Next is Last + N2.

