% Memoize calls to pos (dynamic programming), otherwise
% the program doesn't terminate in reasonable time.
:- table pos/4 as dynamic.

main :-
    % Very useful for debugging!
    %leash(-all), trace,
    read_input(0),
    aggregate_all(max(T), pos(0, T, _, _), FT), !,
    forall(member(Part-Tail, [1-1, 2-9]),
        (aggregate_all(count, X-Y, (between(0, FT, T), pos(Tail, T, X, Y)), Ans),
        format("Part ~d: ~d~n", [Part, Ans]))
    ).

dir("R", 1, 0).
dir("L", -1, 0).
dir("U", 0, 1).
dir("D", 0, -1).

read_input(T) :-
    read_line_to_string(user_input, Line),
    ( Line == end_of_file -> !;
        split_string(Line, " ", "", [D, S]),
        number_string(Cnt, S),
        ST is T + 1,
        NT is T + Cnt,
        dir(D, DX, DY),
        forall(between(ST, NT, CT), (
            PT is CT-1,
            pos(0, PT, PX, PY),
            NX is PX + DX,
            NY is PY + DY,
            assertz(pos(0, CT, NX, NY))
        )),
        read_input(NT)
    ).

pos(_, 0, 0, 0).
pos(I, T, X, Y) :-
    I \= 0, T > 0,
    PT is T-1,
    pos(I, PT, PX, PY),
    H is I-1,
    pos(H, T, HX, HY),
    DX is HX - PX,
    DY is HY - PY,
    ( max(abs(DX), abs(DY)) =< 1 ->
        X = PX, Y = PY ;
        X is PX + sign(DX),
        Y is PY + sign(DY)
    ).

