:- use_module(library(clpfd)).

dig('=', -2).
dig('-', -1).
dig('0', 0).
dig('1', 1).
dig('2', 2).

conv([], 0).
conv([C | Tail], O) :-
	length(Tail, I),
	DV in -2..2,
	O #= V + DV * 5^I,
	conv(Tail, V),
	dig(C, DV).

main :-
	read_string(user_input, _, Data),
	split_string(Data, "\n", "\n", Split),
	maplist(string_chars, Split, CSplit),
	maplist(conv, CSplit, Converted),
	sumlist(Converted, Sum),
	conv(Digs, Sum),
	format("Part 1: ~s~n", [Digs]).
