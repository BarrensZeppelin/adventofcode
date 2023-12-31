:- use_module(library(clpq)).

main :-
    read_string(user_input, _, S), split_string(S, "\n", "\n", Lines),
    maplist([L, Stone]>>(
        re_foldl([re_match{0:N}, [N|Ns], Ns]>>true, "-?\\d+"/t, L, [X,Y,Z|V], [], []),
        pairs_keys_values(Stone, [X,Y,Z], V)
    ), Lines, Stones),

    aggregate_all(count, (
        maplist(slice(2), Stones, Stones2D),
        { T1 >= 0, T2 >= 0 },
        append(L, [A|_], Stones2D), member(B, L),
        maplist({T1,T2}/[P1-V1, P2-V2]>>{
            P =:= P1 + V1 * T1,
            P =:= P2 + V2 * T2,
            200000000000000 =< P, P =< 400000000000000
            % 7 =< P, P =< 27
        }, A, B)
    ), Part1),
    format("Part 1: ~d~n", Part1).

slice(N, L, S) :- length(S, N), append(S, _, L).
