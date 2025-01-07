main :-
    read_string(user_input, _, S), split_string(S, "\n", "\n", Lines),
    maplist([Line, Codes-Bid]>>(
        split_string(Line, " ", "", [CodesS, BidS]),
        string_codes(CodesS, Codes), number_string(Bid, BidS)
    ), Lines, Cards), !,

    solve(Cards, part1, Part1),
    solve(Cards, part2, Part2),
    format("Part 1: ~d~nPart 2: ~d~n", [Part1, Part2]).

strength(part2, 0'J, 0) :- !.
strength(_, Card, Strength) :- string_code(Strength, "23456789TJQKA", Card).

type(part2, [0-Jokers | Tail], Type) :-
    Jokers < 5 ->
    aggregate(max(Cnt, S), member(S-Cnt, Tail), max(Cnt, S)),
    selectchk(S-Cnt, Tail, S-NCnt, Clumped),
    NCnt is Cnt + Jokers,
    type(part1, Clumped, Type).

type(_, Clumped, Type) :- aggregate_all(sum(V*V), member(_-V, Clumped), Type).

solve(Cards, Part, Res) :-
    maplist({Part}/[Cs-Bid, [Typ | Strengths]-Bid]>>(
        maplist(strength(Part), Cs, Strengths),
        msort(Strengths, Sorted), clumped(Sorted, Clumped),
        type(Part, Clumped, Typ)
    ), Cards, Transformed),

    sort(Transformed, Sorted),
    foldl([_-Bid, Id-Acc, NId-R]>>(
        NId is Id + 1,
        R is Acc + Bid * NId
    ), Sorted, 0-0, _-Res).


