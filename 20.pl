:- use_module(library(dcg/basics)), use_module(library(dcg/high_order)).
:- use_module(library(chr)).
:- dynamic adj/2.

str(S) --> string(Cs), { string_codes(S, Cs) }.
module(mod(Name, K, Adj)) -->
    ([K], { memberchk(K, `%&`) }; {K = 0'.}), str(Name), " -> ", sequence(str, ", ", Adj), "\n".

:- chr_constraint counter/2, pulse/1.

counter(P, C), pulse(P) <=> NC is C+1, counter(P, NC).

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

    counter(0, 0), counter(1, 0),
    foreach((between(1, 1000, _), Q = [pulse("button", 0, "broadcaster")|Tail]),
        process(Q-Tail, State)),
    findall(C, current_chr_constraint(counter(_, C)), [L, H]),
    Part1 is L * H,
    format("Part 1: ~d~n", [Part1]).

process([]-_, _).
process([pulse(From, P, To)|Rest]-QT, State) :-
    pulse(P),
    (ht_get(State, To, Spec) ->
        (Spec = fflop(PS) ->
            (P =:= 1 -> NQT = QT;
                NP is 1-PS, ht_put(State, To, fflop(NP)), send(To, NP, QT, NQT));
         Spec = conj(Lows),
            (P =:= 1 -> subtract(Lows, [From], NLows); union(Lows, [From], NLows)),
            (NLows == [] -> NP is 0; NP is 1),
            ht_put(State, To, conj(NLows)),
            send(To, NP, QT, NQT));
        send(To, P, QT, NQT)),
    process(Rest-NQT, State).

send(From, V, QT, NQT) :-
    adj(From, L) ->
        foldl({From,V}/[To, [pulse(From, V, To)|Tail], Tail]>>true, L, QT, NQT);
        QT = NQT.
