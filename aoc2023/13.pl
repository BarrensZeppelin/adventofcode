:- use_module(library(clpfd)).

main :-
    read_string(user_input, _, S), re_split("\n\n", S, Blocks),
    convlist([Block, Codes]>>(
        split_string(Block, "\n", "\n", Lines),
        Lines \= [""],
        maplist(string_codes, Lines, Codes)
    ), Blocks, Input), !,

    aggregate_all(sum(R), (member(Mir, Input), solve(Mir, 0, R)), Part1),
    aggregate_all(sum(R), (member(Mir, Input), solve(Mir, 1, R)), Part2),
    format("Part 1: ~d~nPart 2: ~d~n", [Part1, Part2]).

solve(Input, Diffs, Result) :-
    (find_reflection([],  Input, Diffs, Y) -> Result is Y * 100;
     transpose(Input, Tr), find_reflection([], Tr, Diffs, Result)).

find_reflection(L, R, Diffs, Y) :-
    L \= [], R \= [],
    aggregate_all(count, (
        nth0(I, L, A), nth0(I, R, B),
        nth0(J, A, C1), nth0(J, B, C2),
        C1 \= C2
    ), Diffs),
    length(L, Y), !.

find_reflection(L, [H|T], Diffs, Y) :-
    find_reflection([H|L], T, Diffs, Y).
