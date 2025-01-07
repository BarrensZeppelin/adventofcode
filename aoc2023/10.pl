:- use_module(library(clpfd)).
:- dynamic grid/2.

main :-
    read_string(user_input, _, S), split_string(S, "\n", "\n", Lines),
    forall((nth1(Y, Lines, Line), string_code(X, Line, C)), assertz(grid(X-Y, C))),
    compile_predicates([grid/2]), !,

    grid(Start, 0'S),
    loop(Start, [], Loop),
    length(Loop, LoopLen),
    Part1 is LoopLen // 2,
    format("Part 1: ~d~n", [Part1]),

    aggregate_all(sum(X1 * Y2 - X2 * Y1), nextto(X1-Y1, X2-Y2, [Start|Loop]), SArea),
    Part2 is abs(SArea) // 2 - Part1 + 1,
    format("Part 1: ~d~n", [Part2]).

conn(0'-, -1, 0). conn(0'-, 1, 0).
conn(0'|, 0, -1). conn(0'|, 0, 1).
conn(0'J, 0, -1). conn(0'J, -1, 0).
conn(0'L, 0, -1). conn(0'L, 1, 0).
conn(0'F, 0, 1). conn(0'F, 1, 0).
conn(0'7, 0, 1). conn(0'7, -1, 0).
conn(0'S, DX, DY) :-
    grid(X-Y, 0'S),
    DX #= NX-X, DY #= NY-Y,
    abs(DX) + abs(DY) #= 1,
    label([NX, NY]),
    adj(NX-NY, X-Y).

adj(P, NX-NY) :-
    P = X-Y,
    grid(P, C), conn(C, DX, DY),
    NX is X+DX, NY is Y+DY.

loop(Cur, Loop, Loop) :- grid(Cur, 0'S), Loop \= [], !.
loop(Cur, Stack, NLoop) :-
    (Stack = [] -> true; Stack = [Prev|_], dif(Prev, Next)),
    adj(Cur, Next), loop(Next, [Cur|Stack], NLoop).
