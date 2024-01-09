:- use_module(library(dcg/basics)), use_module(library(dcg/high_order)).
:- use_module(library(chr)).
:- dynamic adj/2.

str(S) --> string(Cs), { string_codes(S, Cs) }.
module(mod(Name, K, Adj)) -->
    ([K], { memberchk(K, `%&`) }; {K = 0'.}), str(Name), " -> ", sequence(str, ", ", Adj), "\n".

:- chr_constraint counter/2, pulse/2.

counter(P, C), pulse(P, _) <=> NC is C+1, counter(P, NC).

main :-
    phrase_from_stream(sequence(module, Mods), user_input),
    forall(member(mod(Name, _, Adj), Mods), assertz(adj(Name, Adj))),
    compile_predicates([adj/2]),

    ht_new(State),
    foreach(member(mod(Name, 0'%, _), Mods), ht_put(State, Name, fflop(0))),
    convlist({State}/[mod(Name, 0'&, _), _]>>(
        findall(P, (adj(P, Adj), memberchk(Name, Adj)), Prev),
        ht_put(State, Name, conj(Prev))
    ), Mods, _), !,

    (part1(State), fail; true),
    part2(State).

part1(State) :-
    counter(0, 0), counter(1, 0),
    foreach(between(1, 1000, _), press(State)),
    findall(C, current_chr_constraint(counter(_, C)), [L, H]),
    Part1 is L * H,
    format("Part 1: ~d~n", [Part1]), !.

press(State) :-
    Q = [pulse("button", 0, "broadcaster")|Tail],
    foldl(process(State), Q, Tail, []).

process(State, pulse(From, P, To), QT, NQT) :-
    pulse(P, From),
    (ht_get(State, To, Spec) ->
        (Spec = fflop(PS) ->
            (P =:= 1 -> NQT = QT;
                NP is 1-PS, ht_put(State, To, fflop(NP)), send(To, NP, QT, NQT));
         Spec = conj(Lows),
            (P =:= 1 -> subtract(Lows, [From], NLows); union(Lows, [From], NLows)),
            (NLows == [] -> NP is 0; NP is 1),
            ht_put(State, To, conj(NLows)), send(To, NP, QT, NQT));
        send(To, P, QT, NQT)).

send(From, V, QT, NQT) :-
    findall(pulse(From, V, To), (adj(From, L), member(To, L)), QT, NQT).

:- chr_constraint presses/1, first_high/2.

presses(X), pulse(_, "button") <=> Y is X+1, presses(Y).
first_high(X, Y), presses(C) \ pulse(1, X) <=> var(Y) | Y = C.
pulse(_, _) <=> true.

part2(State) :-
    presses(0),
    findall(P, (adj(PP, ["rx"]), adj(P, Adj), memberchk(PP, Adj)), Prev),
    maplist([P, H]>>first_high(P, H), Prev, Hs),
    solve(State, Hs).

solve(State, Hs) :- maplist(number, Hs) ->
    foldl([H, A, R]>>(R is H * A), Hs, 1, Part2),
    format("Part 2: ~d~n", [Part2]);
    press(State), solve(State, Hs).
