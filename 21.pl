:- use_module(library(clpfd)).
:- table reach(_,-), dxy/2.
:- dynamic grid/2.

% TODO: Part 2 is slow, but it works. Maybe there's a faster way to count reachable tiles
% at each distance.

main :-
    read_string(user_input, _, S), split_string(S, "\n", "\n", Lines),
    forall((nth1(Y, Lines, Line), string_code(X, Line, C), C \= 0'#), assertz(grid(X, Y))),
    compile_predicates([grid/2]),

    aggregate_all(max(W), grid(W, W), W), nb_setval(dim, W),

    aggregate_all(sum(Size), (between(0, 64, D), D mod 2 =:= 0, reach(D, T), rb_size(T, Size)), Part1),
    format("Part 1: ~d~n", [Part1]),

    findall(R, (
        between(0, 2, T), MD is W // 2 + W * T,
        aggregate_all(sum(Size), (
            between(1, MD, D), D mod 2 =:= MD mod 2, reach(D, Tree), rb_size(Tree, Size)
        ), R),
        writeln(T-R)
    ), Xs),
    % Xs = [3682,32767,90820],

    foreach(nth0(X, Xs, V), (A * X^2 + B * X + C #= V)),
    [A, B, C] ins 0..1000000,
    once(label([A, B, C])),

    Steps is 26501365 // W,
    Part2 is A * Steps^2 + B * Steps + C,
    format("Part 2: ~d~n", [Part2]).

dxy(X, Y) :- abs(X) + abs(Y) #= 1, label([X, Y]).

reach(-1, E) :- rb_empty(E).
reach(0, Tree) :- nb_getval(dim, W), X is (W+1) // 2, list_to_rbtree([X-X-0], Tree).
reach(D, Tree) :-
    D >= 1, D1 is D-1, D2 is D-2, reach(D1, Old), reach(D2, Old2),
    nb_getval(dim, W),
    setof(NX-NY-0, X^Y^DX^DY^(
        rb_in(X-Y, 0, Old), dxy(DX, DY), NX is (X + DX), NY is (Y + DY)
    ), Pairs),
    findall(X-Y-0, (
        member(X-Y-0, Pairs), MX is (X-1) mod W + 1, MY is (Y-1) mod W + 1,
        grid(MX, MY), \+ rb_in(X-Y, _, Old2)
    ), Filtered),
    ord_list_to_rbtree(Filtered, Tree).
