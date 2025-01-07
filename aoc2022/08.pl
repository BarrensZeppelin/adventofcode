:- use_module(library(clpfd)).
:- dynamic(height/3).

main :-
    read_string(user_input, _, Data),
    string_lines(Data, Lines),
    length(Lines, N),

    forall((between(1, N, X), between(1, N, Y)),
        (nth1(Y, Lines, Line),
         string_code(X, Line, C), H is C - 0'0,
         assertz(height(X, Y, H)))
    ), !,

    %leash(-all), trace,
    aggregate_all(count, visible(X, Y, N), P1),
    %findall(true, visible(X, Y, N), Ls),
    %length(Ls, P1),
    format("Part 1: ~d~n", [P1]),
    aggregate_all(max(S), scenic_score(X, Y, N, S), P2),
    format("Part 2: ~d~n", [P2]).

visible(X, Y, N) :-
    height(X, Y, H),
    H2 #< H,
    once(
        X is 1; Y is 1; X is N; Y is N;
        aggregate_all(max(H3), (X2 #< X, height(X2, Y, H3)), H2) ;
        aggregate_all(max(H3), (X2 #> X, height(X2, Y, H3)), H2) ;
        aggregate_all(max(H3), (Y2 #< Y, height(X, Y2, H3)), H2) ;
        aggregate_all(max(H3), (Y2 #> Y, height(X, Y2, H3)), H2)
    ).

scenic_score(X, Y, N, S) :-
    height(X, Y, H),
    %format("~d ~d ~d ~d~n", [X, Y, N, H]),
    (aggregate_all(max(X2), (X2 #< X, H2 #>= H, height(X2, Y, H2)), XL) -> true; XL is 1),
    (aggregate_all(min(X2), (X2 #> X, H2 #>= H, height(X2, Y, H2)), XR) -> true; XR is N),
    (aggregate_all(max(Y2), (Y2 #< Y, H2 #>= H, height(X, Y2, H2)), YL) -> true; YL is 1),
    (aggregate_all(min(Y2), (Y2 #> Y, H2 #>= H, height(X, Y2, H2)), YR) -> true; YR is N),
    S is (X - XL) * (XR - X) * (Y - YL) * (YR - Y).
