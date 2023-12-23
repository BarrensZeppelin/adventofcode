:- use_module(library(dcg/basics)), use_module(library(clpfd)).

list([H|T], Element, Sep) --> call(Element, H), list_(T, Element, Sep).
list_([H|T], Element, Sep) --> Sep, call(Element, H), list_(T, Element, Sep).
list_([], _, _) --> [].

workflow(Name-Rules) --> string_without("{\n", NameC),
    "{", list(Rules, rule, ","), "}", { string_codes(Name, NameC) }.
rule([Key,Op,Val,To]) --> [C, COp], { member(COp, `<>`) }, integer(Val), ":",
    string(ToC), { maplist(char_code, [Key, Op], [C, COp]), string_codes(To, ToC) }.
rule([To]) --> string(ToC), { string_codes(To, ToC) }.

part(Part) --> "{", list(Part, pval, ","), "}".
pval(Key-Val) --> [C, 0'=], integer(Val), { char_code(Key, C) }.

parse(Workflows, Parts) --> list(Workflows, workflow, "\n"), "\n\n", list(Parts, part, "\n"), "\n".

score1(Part, Score) :- aggregate_all(sum(V), member(_-V, Part), Score).

process(_, "A", Part, ScoreFun, Score) :- call(ScoreFun, Part, Score).
process(_, "R", _, _, 0).
process(Workflows, Name, Part, ScoreFun, Res) :-
    memberchk(Name-Rules, Workflows),
    process_rules(Workflows, Rules, Part, ScoreFun, Res).

process_rules(Workflows, [[To]|_], Part, ScoreFun, Res) :-
    process(Workflows, To, Part, ScoreFun, Res).
process_rules(Workflows, [[Key, Op, Val, To]|Tail], Part, ScoreFun, Res) :-
    memberchk(Key-CV, Part),
    (Op = < -> CV #< Val #<==> B; CV #> Val #<==> B),
    aggregate_all(sum(R), (
        (B #= 1, process(Workflows, To, Part, ScoreFun, R));
        (B #= 0, process_rules(Workflows, Tail, Part, ScoreFun, R))
    ), Res).

score2(Part, Score) :- foldl([_-V,A,B]>>(fd_size(V, S), B is A * S), Part, 1, Score).

main :-
    phrase_from_stream(parse(Workflows, Parts), user_input),
    aggregate_all(sum(Score), (
        member(Part, Parts),
        process(Workflows, "in", Part, score1, Score)
    ), Part1),
    format("Part 1: ~d~n", [Part1]),

    length(Vs, 4), Vs ins 1..4000,
    pairs_keys_values(Part, [x,m,a,s], Vs),
    process(Workflows, "in", Part, score2, Part2),
    format("Part 2: ~d~n", [Part2]).


