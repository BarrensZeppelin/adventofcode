main :-
    read_input(L),
    foldl([S, Acc, Res] >> (part1(S, R), Res is Acc + R), L, 0, Part1),
	format("Part 1: ~d~n", [Part1]),
    foldl([S, Acc, Res] >> (part2(S, R), Res is Acc + R), L, 0, Part2),
    format("Part 2: ~d~n", [Part2]).

part1(S, R) :-
    string_codes(S, Codes),
    convlist([C, D] >> (0'0 =< C, C =< 0'9, D is C - 0'0), Codes, Digits),
    score(Digits, R).

score(Digits, R) :-
    [First | _] = Digits,
    last(Digits, Last),
    R is First * 10 + Last.

conv_digit("one", 1).
conv_digit("two", 2).
conv_digit("three", 3).
conv_digit("four", 4).
conv_digit("five", 5).
conv_digit("six", 6).
conv_digit("seven", 7).
conv_digit("eight", 8).
conv_digit("nine", 9).

conv_full([], []).
conv_full(L, Digs) :-
    L = [H | T],
    conv_full(T, Tail),
    (0'0 =< H, H =< 0'9 ->
        D is H - 0'0,
        Digs = [D | Tail];
    conv_digit(Str, D), string_codes(Str, Cs), append(Cs, _, L) ->
        Digs = [D | Tail];
        Digs = Tail).

part2(S, R) :-
    string_codes(S, Codes),
    conv_full(Codes, Digits),
    score(Digits, R).

read_input(L) :-
    read_line_to_string(user_input, Line),
    (Line == end_of_file -> L = [];
        read_input(Tail),
        L = [Line | Tail]
    ).
