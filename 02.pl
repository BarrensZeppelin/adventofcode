main :-
    read_string(user_input, _, Input),
    split_string(Input, "\n", "\n", Lines),
    maplist([L, G]>>(
        re_foldl([re_match{0:_, count:Count, color:Color}, DIn, DOut]>>(
            get_dict(Color, DIn, OldCount),
            (OldCount >= Count -> DOut = DIn; put_dict(Color, DIn, Count, DOut))
        ), "(?<count_I>\\d+) (?<color_A>r|g|b)", L, cnt{r:0, g:0, b:0}, G, [])
    ), Lines, Games),

    foldl([G, Acc-Id, Res-NId]>>(
        NId is Id + 1,
        (maxcount(Color, Max), get_dict(Color, G, Count), Count > Max ->
            Res is Acc; Res is Acc + NId)
    ), Games, 0-0, Part1-_),
	format("Part 1: ~d~n", [Part1]),

    foldl([G, Acc, Res]>>(
        findall(V, get_dict(_, G, V), Vs),
        foldl([V, A, R]>>(R is A * V), Vs, 1, Power),
        Res is Acc + Power
    ), Games, 0, Part2),
    format("Part 2: ~d~n", [Part2]).

maxcount(r, 12).
maxcount(g, 13).
maxcount(b, 14).
