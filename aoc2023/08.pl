:- dynamic adj/3.

main :-
    read_line_to_codes(user_input, Instructions),
    read_string(user_input, _, S),
    re_foldl([re_match{0:_, 1:From, 2:L, 3:R}, _, _]>>(
        assertz(adj(From, 0'L, L)), assertz(adj(From, 0'R, R))
    ), "(\\w+) = \\((\\w+), (\\w+)\\)", S, _, _, []),
    compile_predicates([adj/3]),
    append(Instructions, Tail, Queue),
    path_length(Queue, Tail, "AAA", "ZZZ", Part1),
    format("Part 1: ~d~n", [Part1]),

    findall(N, (adj(Cur, 0'L, _), string_code(3, Cur, 0'A), path_length(Queue, Tail, Cur, "Z", N)), Ls),
    foldl([X, Y, Z]>>(Z is lcm(X, Y)), Ls, 1, Part2),
    format("Part 2: ~d~n", [Part2]).

path_length(_, _, Cur, End, 0) :- sub_string(Cur, _, _, 0, End), !.
path_length([I|Queue], [I|Tail], Cur, End, N) :-
    adj(Cur, I, Next),
    path_length(Queue, Tail, Next, End, NX),
    N is NX + 1.
