:- use_module(library(clpfd)).
:- dynamic prog/2.

eval(I, R, O) :-
    number(I), J is I+1, K is I+2,
    (prog(I, P), prog(J, Q) -> inst(P, K, Q, R, O); O = []).

combo(X, _, X) :- X < 4, !.
combo(4, r(A, _, _), A).
combo(5, r(_, B, _), B).
combo(6, r(_, _, C), C).

inst(0, I, Q, R, O) =>
    R = r(A, B, C),
    combo(Q, R, K), NA #= A >> K, eval(I, r(NA, B, C), O).
inst(1, I, Q, r(A, B, C), O) => NB #= B xor Q, eval(I, r(A, NB, C), O).
inst(2, I, Q, R, O) =>
    R = r(A, _, C),
    combo(Q, R, K), NB #= K mod 8, eval(I, r(A, NB, C), O).
inst(3, I, Q, R, O) =>
    R = r(A, _, _),
    (A #= 0, eval(I, R, O) ; A #> 0, eval(Q, R, O)).
inst(4, I, _, r(A, B, C), O) => NB #= B xor C, eval(I, r(A, NB, C), O).
inst(5, I, Q, R, O) => combo(Q, R, K), J #= K mod 8, O = [J|Tl], eval(I, R, Tl).
inst(6, I, Q, R, O) =>
    R = r(A, _, C),
    combo(Q, R, K), NB #= A >> K, eval(I, r(A, NB, C), O).
inst(7, I, Q, R, O) =>
    R = r(A, B, _),
    combo(Q, R, K), NC #= A >> K, eval(I, r(A, B, NC), O).

main :-
    read_string(user_input, _, S),
    re_foldl([_{0:N}, [N|Ns], Ns]>>true, "\\d+"/t, S, [A, B, C | Prog], [], []),
    foreach(nth0(I, Prog, V), assertz(prog(I, V))),
    eval(0, r(A, B, C), O),
    MA #>= 0,
    eval(0, r(MA, 0, 0), Prog),
    labeling([min(MA), bisect], [MA]),
    format("Part 1: ~w~nPart 2: ~d~n", [O, MA]).
