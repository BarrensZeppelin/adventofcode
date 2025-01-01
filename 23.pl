:- dynamic edge/2.

main :-
    read_string(user_input, _, Input), split_string(Input, "\n", "\n", Lines),
    forall(member(Line, Lines), (
        split_string(Line, "-", "", [A, B]),
        assertz(edge(A, B)), assertz(edge(B, A))
    )),
    aggregate_all(count, (
      edge(A, B), edge(A, C), edge(B, C),
      once((member(S, [A, B, C]), string_code(1, S, 0't)))
    ), Count),
    P1 is Count // 6,
    format("Part 1: ~d\n", [P1]).
