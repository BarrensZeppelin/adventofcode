:- use_module(library(clpfd)).

main :-
    read_input(L),
    part1(L),
    part2(L).

inclusion([L1, R1, L2, R2]) :-
    (L1 #=< L2, R2 #=< R1) ; (L2 #=< L1, R1 #=< R2).

part1(L) :-
    include(inclusion, L, Matches),
    length(Matches, N),
    writeln(N).

overlap([L1, R1, L2, R2]) :-
    inclusion([L1, R1, X, X]), inclusion([L2, R2, X, X]).

part2(L) :-
    include(overlap, L, Matches),
    length(Matches, N),
    writeln(N).

read_input(L) :-
    read_line_to_string(user_input, Line),
    (Line == end_of_file -> L = [];
        term_string(L1-R1','L2-R2, Line),
        %split_string(Line, ",", "", Split),
        %maplist([P, [L, R]]>>term_string(L-R, P), Split, Parsed),
        %append(Parsed, Head),
        read_input(Tail),
        L = [[L1, R1, L2, R2] | Tail]
        %L = [Head | Tail]
    ).
