:- use_module(library(clpq)), use_module(library(clpfd)).

main :-
    read_string(user_input, _, S), split_string(S, "\n", "\n", Lines),
    maplist([L, Stone]>>(
        re_foldl([re_match{0:N}, [N|Ns], Ns]>>true, "-?\\d+"/t, L, [X,Y,Z|V], [], []),
        pairs_keys_values(Stone, [X,Y,Z], V)
    ), Lines, Stones),

    aggregate_all(count, (
        maplist([St,R]>>append(R, [_], St), Stones, Stones2D),
        { T1 >= 0, T2 >= 0 },
        append(L, [A|_], Stones2D), member(B, L),
        maplist({T1,T2}/[P1-V1, P2-V2]>>{
            P =:= P1 + V1 * T1,
            P =:= P2 + V2 * T2,
            200000000000000 =< P, P =< 400000000000000
            % 7 =< P, P =< 27
        }, A, B)
    ), Part1),
    format("Part 1: ~d~n", Part1),

    Params = [X-XV, Y-YV, Z-ZV],
    [XV,YV,ZV] ins -3..300,
    X #>= 0, Y #>= 0, Z #>= 0,
    maplist({Params}/[St, T]>>(
        T #> 0,
        maplist({T}/[P1-V1, P2-V2]>>(
            P1 #= P2 + (V2 - V1) * T
            % P1 + V1 * T #= P2 + V2 * T
        ), St, Params)
    ), Stones, Times),
    once(label([X,Y,Z,XV,YV,ZV|Times])),
    Part2 is X+Y+Z,
    format("Part 2: ~d~n", Part2).
