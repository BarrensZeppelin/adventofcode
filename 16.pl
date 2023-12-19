:- dynamic grid/3.

main :-
    read_string(user_input, _, S), split_string(S, "\n", "\n", Lines),
    forall((nth1(Y, Lines, Line), string_code(X, Line, C)), assertz(grid(X, Y, C))),
    compile_predicates([grid/3]),

    count_reachable(1-1, 0, Part1),
    format("Part 1: ~d~n", [Part1]),

    aggregate_all(max(X), grid(X, _, _), D),
    aggregate_all(max(Count), (
        startdir(X-Y, D, Dir),
        between(1, D, X), between(1, D, Y),
        count_reachable(X-Y, Dir, Count)
    ), Part2),
    format("Part 2: ~d~n", [Part2]).

startdir(1-_, _, 0). startdir(_-1, _, 3).
startdir(D-_, D, 2). startdir(_-D, D, 1).

count_reachable(X-Y, Dir, Count) :-
    empty_nb_set(Visited),
    reachable(X-Y, Dir, Visited),
    aggregate_all(count, P, gen_nb_set(Visited, P-_), Count).

reachable(P, Dir, Vis) :-
    add_nb_set(P-Dir, Vis, New),
    (New == false -> true;
        forall(move(P, Dir, NP, NDir), reachable(NP, NDir, Vis))).

dxy(D, P) :- nth0(D, [1-0, 0-(-1), -1-0, 0-1], P).

move(X-Y, Dir, NX-NY, NDir) :-
    grid(X, Y, C),
    (string_code(I, "\\/", C) -> (Dir mod 2 =:= I mod 2 -> DD is 1; DD is -1);
     string_code(I, "-|", C), Dir mod 2 =:= I mod 2 -> (DD is 1; DD is -1);
     DD is 0),
    NDir is (Dir + DD) mod 4,
    dxy(NDir, Dx-Dy),
    NX is X + Dx, NY is Y + Dy,
    grid(NX, NY, _).
