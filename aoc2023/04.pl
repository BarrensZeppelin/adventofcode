:- dynamic card/2.
:- table count(_,-).

get_numbers(L, Ns) :- re_foldl([re_match{0:S}, [S|T], T]>>true, "\\d+", L, Ns, [], [capture_type(term)]).

main :-
    read_string(user_input, _, S), split_string(S, "\n", "\n", Lines),
    forall(member(L, Lines), (
        split_string(L, "|", " ", Parts),
        maplist(get_numbers, Parts, [[ID | Win], Mine]),
        aggregate_all(count, (member(N, Mine), memberchk(N, Win)), R),
        assertz(card(ID, R))
    )),

    aggregate_all(sum(2^(R-1)), (card(_, R), R > 0), Part1),
    aggregate_all(sum(C), count(ID, C), Part2),
	format("Part 1: ~d~nPart 2: ~d~n", [Part1, Part2]).

count(ID, Res) :-
    card(ID, R), Lo is ID + 1, Hi is ID + R,
    aggregate_all(sum(C), (C is 1; between(Lo, Hi, N), count(N, C)), Res).
