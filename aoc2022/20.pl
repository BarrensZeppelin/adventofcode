mix(0, _, L) :- printans(L).
mix(Iters, Order, IL) :-
	smix(Order, IL, OL),
	NIters is Iters - 1,
	mix(NIters, Order, OL).

smix([], L, L).
smix([P | Tail], IL, OL) :-
	nth0(I, IL, P, Rest),
	_-Change = P,
	length(IL, N),
	J is (I + Change) mod (N-1),
	nth0(J, ML, P, Rest),
	% Discard choice points to save stack memory!
	!,
	smix(Tail, ML, OL).


printans(L) :-
	length(L, N),
	nth0(I, L, _-0),
	aggregate_all(sum(C), (between(1, 3, X), J is (I+(X*1000)) mod N, nth0(J, L, _-C)), Ans),
	writeln(Ans).

main :-
	read_string(user_input, _, Data),
	split_string(Data, "\n", "\n", Split),
	length(Split, N),
	numlist(1, N, Ind),
	maplist([S, I, I-V]>>number_string(V, S), Split, Ind, L),
	mix(1, L, L),
	maplist([I-V, I-V2]>>(V2 is V * 811589153), L, L2),
	mix(10, L2, L2).
