:- dynamic grid/3, edge/3, dfs/3.

% Essentially just a translation of the Python solution to Prolog. 🤔

main :-
    read_string(user_input, _, S),
    forall((
        split_string(S, "\n", "\n", Lines),
        nth1(Y, Lines, Line), string_code(X, Line, C),
        C \= 0'#
    ), assertz(grid(X, Y, C))),
    compile_predicates([grid/3]),

    aggregate_all(r(max(X), max(Y)), grid(X, Y, _), r(W, H)),
    findall(X-Y, (
        grid(X, Y, 0'.),
        aggregate_all(count, adj(X-Y, _), C), C >= 3
    ), Rest),
    Waypoints = [2-1, W-H | Rest],

    forall((
        nth0(I, Waypoints, P), adj(P, NP),
        dif(P, NP2), once(adj(NP, NP2)),
        find_path(NP, P, Waypoints, R-P2)
    ), (nth0(J, Waypoints, P2), assertz(edge(I, J, R)))),
    once(edge(X, 1, D)), asserta((dfs(X, _, D) :- !)),
    compile_predicates([dfs/3]),

    dfs(0, 0, Part1),
    format("Part 1: ~d~n", [Part1]),

    findall(edge(B, A, C), edge(A, B, C), REdges),
    maplist(assertz, REdges),
    compile_predicates([edge/3]),
    dfs(0, 0, Part2),
    format("Part 2: ~d~n", [Part2]).

adj(X-Y, NX-NY) :-
    grid(X, Y, C), dxy(C, DX, DY),
    NX is X+DX, NY is Y+DY, grid(NX, NY, _).

find_path(P, _, WP, 1-P) :- memberchk(P, WP), !.
find_path(P, PP, WP, R-RP) :-
    adj(P, NP), NP \= PP, find_path(NP, P, WP, R0-RP), R is R0+1.

dxy(0'., X, Y) :- dif(C, 0'.), dxy(C, X, Y).
dxy(C, X, Y) :- member(C-X-Y, [0'>-1-0, 0'v-0-1, 0'<-(-1)-0, 0'^-0-(-1)]).

dfs(P, Vis, Res) :-
    getbit(Vis, P) =:= 0, NVis is Vis \/ (1<<P),
    aggregate_all(max(R), (
        edge(P, NP, W), dfs(NP, NVis, R0), R is R0+W
    ), Res).
