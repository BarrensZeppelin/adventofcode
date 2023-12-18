main :-
    read_string(user_input, _, S), split_string(S, "\n", "\n", Lines),

    maplist([Line, D-Steps]>>(
        split_string(Line, " ", "", [D, StepsStr, _]),
        number_string(Steps, StepsStr)
    ), Lines, Ins1),
    solve(Ins1, Part1),

    maplist([Line, D-Steps]>>(
        re_matchsub("#(\\w{5})(\\w)", Line, _{0:_, 1:StepsStr, 2:DS}, []),
        number_string(DSI, DS), sub_string("RDLU", DSI, 1, _, D),
        string_concat("0x", StepsStr, StepsHex), number_string(Steps, StepsHex)
    ), Lines, Ins2),
    solve(Ins2, Part2),
    format("Part 1: ~d~nPart 2: ~d~n", [Part1, Part2]).

dx(C, D) :- string_code(I, "RULD", C), nth1(I, [1, 0, -1, 0], D).
dy(C, D) :- string_code(I, "RULD", C), nth1(I, [0, 1, 0, -1], D).

solve(Instructions, Result) :-
    foldl([D-Steps, X-Y-Area-Side, NX-NY-NArea-NSide]>>(
        dx(D, Dx), dy(D, Dy),
        NX is X + Dx * Steps, NY is Y + Dy * Steps,
        NSide is Side + Steps, NArea is Area + X * NY - Y * NX
    ), Instructions, 0-0-0-0, _-_-Area-Side),
    Int is abs(Area) // 2 - Side // 2 + 1,
    Result is Int + Side.



