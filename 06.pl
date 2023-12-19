:- use_module(library(clpfd)).

main :-
    read_string(user_input, _, S), split_string(S, "\n", "\n", Lines),
    maplist({Lines}/[Parse, Res]>>(maplist(Parse, Lines, Input), solve(Input, Res)), [
        [Line, Ns]>>re_foldl([_{0:N}, [N|T], T]>>true, "\\d+"/t, Line, Ns, [], []),
        [Line, [V]]>>re_foldl([_{0:N}, A, B]>>(B is A*10+N), "\\d"/t, Line, 0, V, [])
    ], Ans),
    format("Part 1: ~d~nPart 2: ~d~n", Ans).

solve([[], []], 1).
solve([[T|Ts], [D|Ds]], Res) :-
    P #< T, P * (T - P) #> D, fd_size(P, W),
    solve([Ts, Ds], Res1),
    Res is Res1 * W.
