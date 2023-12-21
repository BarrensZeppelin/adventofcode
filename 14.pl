:- use_module(library(clpfd)), use_module(library(prolog_stack)).
:- table cycle(_,-), process(_,_) as subsumptive.

main :-
    % open("14.in", read, Stream),
    Stream = user_input,
    read_string(Stream, _, S), split_string(S, "\n", "\n", Lines),
    maplist(string_codes, Lines, Codes),
    rotate(Codes, Input, ccw),

    part1(Input, Part1),
    format("Part 1: ~d~n", [Part1]),

    part2(Input, Part2),
    format("Part 2: ~d~n", [Part2]).

rotate(Matrix, Rotated, Dir) :-
    transpose(Matrix, Transposed),
    (Dir = ccw -> reverse(Transposed, Rotated);
        maplist(reverse, Transposed, Rotated)).

part1(Input, Res) :- tilt(Input, Tilted), load(Tilted, Res).

tilt(Input, Tilted) :- maplist(tilt_row, Input, Tilted).
tilt_row(Row, Tilted) :- tilt_row(0, Row, Tilted).
tilt_row(Free, [0'#|Row], Tilted) :-
    make_dots(Free, Dots), tilt_row(0, Row, Tail),
    append([Dots, `#`, Tail], Tilted).
tilt_row(Free, [0'O|Row], [0'O|Tail]) :- tilt_row(Free, Row, Tail).
tilt_row(Free, [0'.|Row], Tilted) :- NFree is Free+1, tilt_row(NFree, Row, Tilted).
tilt_row(Free, [], Dots) :- make_dots(Free, Dots).

make_dots(N, Dots) :- length(Dots, N), maplist(=(0'.), Dots).

load(Rows, Load) :-
    aggregate_all(sum(Y), (member(Row, Rows), reverse(Row, Rev), nth1(Y, Rev, 0'O)), Load).

cycle(Input, Output) :- cycle(Input, Output, 4).
cycle(Input, Input, 0).
cycle(Input, Output, N) :-
    N > 0, N1 is N-1,
    tilt(Input, Tilted),
    rotate(Tilted, Rotated, cw),
    cycle(Rotated, Output, N1).

:- dynamic(in/1).

process(0, Input) :- in(Input).
process(N, Output) :-
    N > 0, N1 is N-1,
    process(N1, Processed),
    cycle(Processed, Output).

part2(Input, Res) :-
    assertz(in(Input)),

    NumCycles is 1000000000,
    between(1, NumCycles, D1),
    process(D1, Head),
    MaxD2 is D1-1, between(1, MaxD2, D2),
    process(D2, Head),
    Steps is D2 + (NumCycles-D2) mod (D1-D2),
    process(Steps, Final),
    load(Final, Res).
