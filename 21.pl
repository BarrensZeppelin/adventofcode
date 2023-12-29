:- use_module(library(clpfd)).
:- table reach(_,_,_,min), dxy/2.
:- dynamic grid/2.

main :-
    open("21.in", read, Stream),
    read_string(Stream, _, S), split_string(S, "\n", "\n", Lines),
    forall((nth1(Y, Lines, Line), string_code(X, Line, C), C \= 0'#), assertz(grid(X, Y))),
    compile_predicates([grid/2]),

    aggregate_all(max(W), grid(W, W), W), nb_setval(dim, W),

    count_reachable(64, Part1),
    format("Part 1: ~d~n", [Part1]),

    findall(R, (
        between(0, 2, T), MD is W // 2 + W * T,
        count_reachable(MD, R),
        writeln(T-R)
    ), Xs),
    % Xs = [3682,32768,90820],

    foreach(nth0(X, Xs, V), (A * X^2 + B * X + C #= V)),
    [A, B, C] ins 0..1000000,
    once(label([A, B, C])),

    Steps is 26501365 // W,
    Part2 is A * Steps^2 + B * Steps + C,
    format("Part 2: ~d~n", [Part2]).

dxy(X, Y) :- abs(X) + abs(Y) #= 1, label([X, Y]).

count_reachable(Hi, R) :-
    abolish_table_subgoals(reach/4),
    aggregate_all(count, (reach(_, _, Hi, D), D mod 2 =:= Hi mod 2), R).

reach(X, X, _, 0) :- nb_getval(dim, W), X is (W+1) // 2.
reach(X, Y, Hi, D) :- nb_getval(dim, W),
    reach(A, B, Hi, D1), D1 < Hi,
    dxy(DX, DY), X is A + DX, Y is B + DY,
    MX is (X-1) mod W + 1, MY is (Y-1) mod W + 1,
    grid(MX, MY), D is D1+1.
