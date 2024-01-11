% Straightforward (but unsatisfyingly procedural) implementation of Karger's algorithm.
% The union find datastructure uses Prolog's internal unification machinery, which
% is efficient but quite hacky.

main :-
    % set_random(seed(0)),
    read_string(user_input, _, S),
    findall(X-Y, (
        split_string(S, "\n", "\n", Lines), member(Line, Lines),
        split_string(Line, " ", ":", Ns), maplist(atom_string, [X|Ys], Ns),
        member(Y, Ys)
    ), Edges),
    setof(X, Y^(member(X-Y, Edges); member(Y-X, Edges)), Nodes),
    solve(Edges, Nodes).

solve(Edges, Nodes) :-
    length(Nodes, N),
    maplist([X,X-_]>>true, Nodes, Pairs), dict_create(P, _, Pairs),
    random_permutation(Edges, Perm),
    contract(P, Perm, N),
    aggregate_all(count, (
        member(A-B, Edges), get_dict(A, P, X), get_dict(B, P, Y), X \== Y
    ), Cut),
    (Cut > 3 -> solve(Edges, Nodes);
        get_dict(_, P, X), $,
        aggregate_all(count, (get_dict(_, P, Y), Y == X), C),
        Ans is C * (N-C),
        writeln(Ans)).

contract(_, _, 2) :- !.
contract(P, [A-B|Tail], N) :-
    get_dict(A, P, X), get_dict(B, P, Y),
    (X == Y -> NN = N; NN is N-1, X = Y),
    contract(P, Tail, NN).
