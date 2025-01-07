:- [library(dcg/basics), library(dcg/high_order)].
:- [library(chr), library(aggregate), library(apply), library(apply_macros), library(yall)].
:- dynamic adj/2.

str(S) --> string(Cs), { string_codes(S, Cs) }.
module(mod(Name, K, Adj)) -->
    ([K], { memberchk(K, `%&`) }; {K = 0'.}), str(Name), " -> ", sequence(str, ", ", Adj), "\n".

:- chr_constraint counter(+, +), pulse(+, +).

counter(P, C), pulse(P, _) <=> NC is C+1, counter(P, NC).

init_state(_, 0'%, fflop(0)).
init_state(Name, 0'&, conj(Prev)) :- findall(P, (adj(P, Adj), memberchk(Name, Adj)), Prev).
init_state(_, 0'., normal).

main :-
    open("20.in", read, Stream),
    phrase_from_stream(sequence(module, Mods), Stream), !,
    forall(member(mod(Name, _, Adj), Mods), assertz(adj(Name, Adj))),
    compile_predicates([adj/2]),

    maplist([mod(Name, K, _), Name-Spec]>>once(init_state(Name, K, Spec)), Mods, Pairs),
    ht_pairs(State, Pairs),

    ($(part1(State)), fail; true),
    profile(part2(State)).

part1(State) =>
    counter(0, 0), counter(1, 0),
    foreach(between(1, 1000, _), press(State)),
    findall(C, current_chr_constraint(counter(_, C)), [L, H]),
    Part1 is L * H,
    format("Part 1: ~d~n", [Part1]).

press(State) :-
    foldl({State}/[pulse(From, P, To), QT, NQT]>>(
        pulse(P, From),
        ht_put(State, To, NS, normal, OS),
        (process(OS, From, P, NS, NP) ->
            findall(pulse(To, NP, NTo), (adj(To, L), member(NTo, L)), QT, NQT)
        ; QT = NQT, OS = NS)
    ), [pulse("button", 0, "broadcaster")|Tail], Tail, []), !.

process(normal, _, P, normal, P).
process(fflop(P), _, 0, fflop(NP), NP) :- NP is 1-P.
process(conj(Lows), From, P, conj(NLows), NP) :-
    (P =:= 1 -> subtract(Lows, [From], NLows); union(Lows, [From], NLows)),
    (NLows == [] -> NP is 0; NP is 1).

:- chr_constraint presses(+natural), first_high(+, -).

presses(X), pulse(_, "button") <=> Y is X+1, presses(Y).
presses(C) \ first_high(X, Y), pulse(1, X) <=> Y = C.
pulse(_, _) <=> true.

part2(State) :-
    presses(0),
    findall(P, (adj(PP, ["rx"]), adj(P, Adj), memberchk(PP, Adj)), Prev),
    maplist(first_high, Prev, Hs), solve(State, Hs).

solve(State, Hs) :- maplist(number, Hs) ->
    foldl([H, A, R]>>(R is H * A), Hs, 1, Part2),
    format("Part 2: ~d~n", [Part2]);
    press(State), solve(State, Hs).
