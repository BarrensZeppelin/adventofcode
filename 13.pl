:- use_module([library(clpq), library(yall), library(apply_macros)]).

main :-
    read_string(user_input, _, S), re_split("\n\n", S, Sections),
    maplist([S, L]>>re_foldl([_{0:N}, [N|Ns], Ns]>>true, "\\d+"/t, S, L, [], []), Sections, Inputs),
    maplist({Inputs}/[X, Ans]>>
        aggregate_all(sum(3 * A + B), (
            member([Ax, Ay, Bx, By, Px, Py], Inputs), {
                A >= 0, B >= 0,
                Px + X = Ax * A + Bx * B,
                Py + X = Ay * A + By * B
            }, integer(A), integer(B)
        ), Ans), [0, 10^13], Ans),
    format("Part 1: ~d~nPart 2: ~d~n", Ans).
