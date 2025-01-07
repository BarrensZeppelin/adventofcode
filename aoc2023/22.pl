:- use_module(library(dcg/basics)), use_module(library(dcg/high_order)).
:- dynamic edge/2, pt/4.

brick(Brick) --> sequence(sequence(integer, ","), "~", [L, H]), "\n",
    { pairs_keys_values(Brick, L, H) }.

main :-
    phrase_from_stream(sequence(brick, Bricks), user_input),
    predsort(cmp, Bricks, Sorted),
    forall(nth0(I, Sorted, B), (
        (aggregate_all(max(Z), (area(B, X, Y), pt(X, Y, Z, _)), NZ) ->
            forall(distinct(J, (area(B, X, Y), pt(X, Y, NZ, J))), assertz(edge(I, J)));
            NZ = 0),
        [_, _, ZL-ZH] = B,
        Z is NZ + 1 + ZH - ZL,
        forall(area(B, X, Y), assertz(pt(X, Y, Z, I)))
    )),

    aggregate_all(count, (
        nth0(I, Sorted, _), \+ (edge(J, I), dif(I, K), \+ edge(J, K))
    ), Part1),
    format("Part 1: ~d~n", [Part1]),

    aggregate_all(sum(Size-1), (
        nth0(I, Sorted, _),
        ht_new(V), solve([I], V), ht_size(V, Size)
    ), Part2),
    format("Part 2: ~d~n", [Part2]).

get_z([_,_,Z-_], Z).
cmp(R, A, B) :- get_z(A, X), get_z(B, Y), (X =< Y -> R = <; R = >).

area([XL-XH, YL-YH, _], X, Y) :- between(XL, XH, X), between(YL, YH, Y).

solve([], _) :- !.
solve([I|Tail], Deleted) :-
    ht_put(Deleted, I, 0),
    findall(J, (edge(J, I), forall(edge(J, K), ht_get(Deleted, K, 0))), Q, Tail),
    solve(Q, Deleted).
