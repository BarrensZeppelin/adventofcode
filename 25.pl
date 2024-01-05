:- dynamic parent/2.

% Straightforward (but unsatisfyingly procedural) implementation of Karger's algorithm.

main :-
    % set_random(seed(0)),
    read_string(user_input, _, S),
    findall(X-Y, (
        split_string(S, "\n", "\n", Lines), member(Line, Lines),
        split_string(Line, " ", ":", [X|Ys]),
        member(Y, Ys)
    ), Edges),
    setof(X, Y^(member(X-Y, Edges); member(Y-X, Edges)), Nodes),
    solve(Edges, Nodes).

solve(Edges, Nodes) :-
    length(Nodes, N),
    retractall(parent(_, _)),
    forall(member(X, Nodes), assertz(parent(X, X))),
    random_permutation(Edges, Perm),
    contract(Perm, N),
    aggregate_all(count, (
        member(A-B, Edges), find(A, X), find(B, Y), X \= Y
    ), Cut),
    (Cut > 3 -> solve(Edges, Nodes);
        [A-_|_] = Edges,
        find(A, X),
        aggregate_all(count, (member(B, Nodes), find(B, X)), C),
        Ans is C * (N-C),
        writeln(Ans)).

find(X, Y) :-
    parent(X, Z),
    (X == Z -> Y = X; parent(Z, Z) -> Y = Z;
     find(Z, Y), retract(parent(X, Z)), assertz(parent(X, Y))).

contract(_, 2) :- !.
contract([A-B|Tail], N) :-
    find(A, X), find(B, Y),
    (X = Y -> NN = N; NN is N-1, retract(parent(X, X)), assertz(parent(X, Y))),
    contract(Tail, NN).
