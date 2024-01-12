% Straightforward (but unsatisfyingly procedural) implementation of Karger's algorithm.
% The union find datastructure uses Prolog's internal unification machinery, which
% is efficient but quite hacky.

main :-
    read_string(user_input, _, S),
    findall(X-Y, (
        split_string(S, "\n", "\n", Lines), member(Line, Lines),
        split_string(Line, " ", ":", Ns), maplist(atom_string, [X|Ys], Ns),
        member(Y, Ys)
    ), Edges),
    setof(X, Y^(member(X-Y, Edges); member(Y-X, Edges)), Nodes),
    length(Goals, 4), maplist(=(solve(Edges, Nodes, Ans)), Goals),
    first_solution(Ans, Goals, []), writeln(Ans).

solve(Edges, Nodes, Ans) =>
    length(Nodes, N), % set_random(seed(0)),
    pairs_keys_values(Pairs, Nodes, VNodes), dict_pairs(D, _, Pairs),
    maplist({D}/[A-B, X-Y]>>(dict_create(S, _, [A-X, B-Y]), S :< D), Edges, VEdges),
    repeat, % set-up done, try contracting in random orders
    random_permutation(VEdges, Perm), contract(Perm, N),
    aggregate_all(count, (member(A-B, Perm), A \== B), 3) ->
    aggregate_all(count, ([A|_] = VNodes, member(B, VNodes), A == B), C),
    Ans is C * (N-C).

contract(_, 2) => !.
contract([X-Y|Tail], N) =>
    (X == Y -> NN = N; NN is N-1, X = Y),
    contract(Tail, NN).
