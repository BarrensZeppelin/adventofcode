:- use_module(library(clpfd)), use_module(library(prolog_stack)).

get_numbers(L, Ns) :- re_foldl([re_match{0:S}, [S|T], T]>>true, "\\d+", L, Ns, [], [capture_type(term)]).

main :-
    read_string(user_input, _, S), re_split("\n\n", S, [SeedsS | Blocks]),
    get_numbers(SeedsS, Seeds),
    convlist([Block, Map]>>(
        split_string(Block, "\n", "\n", [_ | Lines]), Lines \= [],
        maplist([L, [SourceStart, SourceEnd, Shift]]>>(
            get_numbers(L, [DestStart, SourceStart, Length]),
            SourceEnd is SourceStart + Length,
            Shift is DestStart - SourceStart
        ), Lines, Map)
    ), Blocks, Maps),
    !,
    solve(Maps, Seeds, Part1),
    format("Part 1: ~d\n", [Part1]),

    range_pairs(Seeds, SeedRanges),
    solve(Maps, SeedRanges, Part2),
    format("Part 2: ~d\n", [Part2]).

constrain(Ranges, In, Out) :-
    foldl({In, Out}/[[SourceStart, SourceEnd, Shift], AIn, AOut]>>(
        End is SourceEnd-1,
        (In in SourceStart..End) #<==> B,
        AOut #= (AIn * (1 - B)),
        B #==> (Out #= In + Shift)
    ), Ranges, 1, AllFail),
    AllFail #==> (Out #= In).

solve(Maps, Domains, Out) :-
    domain_union(Initial, Domains),
    scanl(constrain, Maps, Initial, Vars),
    Vars ins 0..10000000000,
    last(Vars, Out),
    once(labeling([min(Out), bisect], Vars)).

domain_union(V, [D|Ds]) :-
    foldl([Dom, DIn, DOut]>>(DOut = DIn \/ Dom), Ds, D, DUnion),
    V in DUnion.

range_pairs([], []).
range_pairs([Seed, Length | Tail], [Seed..End | Pairs]) :-
    End is Seed + Length - 1, range_pairs(Tail, Pairs).
