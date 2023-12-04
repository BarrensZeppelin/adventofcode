:- use_module(library(clpfd)).
:- dynamic grid/2, symb_neighbours/2.

neighbours(Start-Len-Y, SX-SY, C) :-
    abs(Y - SY) #=< 1, SX #>= Start, SX #=< Start+Len+1,
    label([SX, SY]), grid(SX-SY, C).

main :-
    read_string(user_input, _, S), split_string(S, "\n", "\n", Lines),
    foldl([L, Y, NY]>>(
        NY is Y + 1,
        forall((string_code(X, L, C), code_type(C, punct), C \= 0'.), assertz(grid(X-Y, C)))
    ), Lines, 0, _),
    compile_predicates([grid/2]),
    foldl([L, Y, NY]>>(
        NY is Y + 1,
        re_foldl([re_match{0:Start-Len}, _, _]>>(
            sub_string(L, Start, Len, _, Sub), number_string(N, Sub),
            forall(neighbours(Start-Len-Y, SX-SY, _), assertz(symb_neighbours(SX-SY, Y-Start-N)))
        ), "\\d+", L, _, _, [capture_type(range)])
    ), Lines, 0, _),
    aggregate(sum(N), Y-Start, P^symb_neighbours(P, Y-Start-N), Part1),
    format("Part 1: ~d~n", [Part1]),

    aggregate_all(sum(Prod), (grid(X-Y, 0'*), findall(N, symb_neighbours(X-Y, _-_-N), [A, B]), Prod is A*B), Part2),
    format("Part 2: ~d~n", [Part2]).
