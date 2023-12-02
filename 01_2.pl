main :-
    read_input(L),
    foldl([S, Acc, Res] >> (solve(S, R, part1), Res is Acc + R), L, 0, Part1),
	format("Part 1: ~d~n", [Part1]),
    foldl([S, Acc, Res] >> (solve(S, R, part2), Res is Acc + R), L, 0, Part2),
    format("Part 2: ~d~n", [Part2]).

solve(S, R, Part) :-
    first_digit(S, D1, Part),
    last_digit(S, D2, Part),
    R is D1 * 10 + D2.

first_digit(S, D, Part) :-
    string_concat(_, R, S),
    conv_digit(SDigit, D, Part),
    string_concat(SDigit, _, R).

last_digit(S, D, Part) :-
    string_length(S, Len),
    between(1, Len, SLen),
    string_concat(_, R, S),
    string_length(R, SLen),
    conv_digit(SDigit, D, Part),
    string_concat(SDigit, _, R).

conv_digit("one", 1, part2).
conv_digit("two", 2, part2).
conv_digit("three", 3, part2).
conv_digit("four", 4, part2).
conv_digit("five", 5, part2).
conv_digit("six", 6, part2).
conv_digit("seven", 7, part2).
conv_digit("eight", 8, part2).
conv_digit("nine", 9, part2).
conv_digit(S, D, _) :-
    between(0'0, 0'9, C),
    string_codes(S, [C]),
    D is C - 0'0.

read_input(L) :-
    read_line_to_string(user_input, Line),
    (Line == end_of_file -> L = [];
        read_input(Tail),
        L = [Line | Tail]
    ).
