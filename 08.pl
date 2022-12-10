:- use_module(library(clpfd)).
:- dynamic(height/3).

main :-
    %leash(-all), trace,
    read_string(user_input, _, Data),
    string_lines(Data, Lines),
    length(Lines, N),

    forall((between(1, N, X), between(1, N, Y)),
        (nth1(Y, Lines, Line),
         string_code(X, Line, C), H is C - 0'0,
         assertz(height(X, Y, H)))
    ),

    aggregate_all(count, (between(1, N, X), between(1, N, Y), visible(X, Y, N)), P1),
    writeln(P1).

visible(X, Y, N) :-
    height(X, Y, H),
    H2 #< H,
    (
        X is 1; Y is 1; X is N; Y is N;
        aggregate_all(max(H3), (X2 #< X, height(X2, Y, H3)), H2) ;
        aggregate_all(max(H3), (X2 #> X, height(X2, Y, H3)), H2) ;
        aggregate_all(max(H3), (Y2 #< Y, height(X, Y2, H3)), H2) ;
        aggregate_all(max(H3), (Y2 #> Y, height(X, Y2, H3)), H2)
    ), !.
