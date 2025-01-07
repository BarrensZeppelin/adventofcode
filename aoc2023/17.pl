:- dynamic grid/3.
:- table path(_,_,_,_,min).

main :-
    read_string(user_input, _, S), split_string(S, "\n", "\n", Lines),
    forall((nth1(Y, Lines, Line), string_code(X, Line, C), V is C - 0'0),
           assertz(grid(X, Y, V))),
    compile_predicates([grid/3]),

    % Assert that the grid is a square
    aggregate_all(r(max(X), max(Y)), grid(X, Y, _), r(W, W)),
    aggregate_all(min(C), path(W, W, _, 1-3, C), Part1),
    aggregate_all(min(C), path(W, W, _, 4-10, C), Part2),
    format("Part 1: ~d~nPart 2: ~d~n", [Part1, Part2]).

path(1, 1, _, _, 0).
path(X, Y, St, Lo-Hi, R) :-
    (Sign = 1; Sign = -1),
    (St = 0; St = 1), DX is St * Sign, DY is (1-St) * Sign,
    between(Lo, Hi, D),
    NX is X + DX * D, NY is Y + DY * D,
    grid(NX, NY, _), NSt is 1-St, Lim is D-1,
    path(NX, NY, NSt, Lo-Hi, R1),
    aggregate_all(sum(V), (
        between(0, Lim, T),
        TX is X + DX * T, TY is Y + DY * T,
        grid(TX, TY, V)
    ), R2),
    R is R1 + R2.
