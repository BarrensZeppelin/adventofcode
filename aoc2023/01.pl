main :-
    read_string(user_input, _, Input), split_string(Input, "\n", "\n", L),
    maplist(score, [part1, part2], [L, L], Ans),
    format("Part 1: ~d~nPart 2: ~d~n", Ans).

score(Part, Lines, Res) :-
    aggregate_all(sum(D1 * 10 + D2), (
        member(L, Lines),
        aggregate_all(r(min(I, D), max(I, D)), (
            valid_number(Part, Sub, D), sub_string(L, I, _, _, Sub)
        ), r(min(_, D1), max(_, D2)))
    ), Res).

valid_number(_, Sub, I) :- sub_string("_123456789", I, 1, _, Sub).
valid_number(part2, Sub, I) :-
    nth1(I, ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"], Sub).
