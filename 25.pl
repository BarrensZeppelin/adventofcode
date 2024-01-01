% Straightforward (but unsatisfyingly procedural) implementation of Karger's algorithm.

main :-
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
    random_permutation(Edges, Perm),
    ht_new(Parent),
    contract(Perm, Parent, N),
    aggregate_all(count, (
        member(A-B, Edges), find(A, X, Parent), find(B, Y, Parent), X \= Y
    ), Cut),
    (Cut > 3 -> solve(Edges, Nodes);
        [A-_|_] = Edges,
        find(A, X, Parent),
        aggregate_all(count, (member(B, Nodes), find(B, X, Parent)), C),
        Ans is C * (N-C),
        writeln(Ans)).

find(X, Y, Parent) :-
    (ht_get(Parent, X, P) -> find(P, Y, Parent), ht_put(Parent, X, Y);
        X = Y).

contract(_, _, 2) :- !.
contract([A-B|Tail], Parent, N) :-
    find(A, X, Parent), find(B, Y, Parent),
    (X = Y -> NN = N; NN is N-1, ht_put(Parent, X, Y)),
    contract(Tail, Parent, NN).
