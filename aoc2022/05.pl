main :-
    read_string(user_input, _, Data),
    re_split("\n\n", Data, L),
    maplist([S, OL]>>split_string(S, "\n", "\n", OL), L, [StackSpecWE, _, MoveSpec]),
    append(StackSpec, [_], StackSpecWE),
    numlist(0, 8, Ind),
    maplist({StackSpec}/[I, O]>>(
        convlist({I}/[S, C]>>(
            J is 4*I+2, string_code(J, S, C), 0' \= C
        ), StackSpec, O)
    ), Ind, Stacks),
    maplist([S, O]>>(
        split_string(S, " ", "", SS),
        convlist([X, Y]>>number_string(Y, X), SS, O)
    ), MoveSpec, Moves),

    crane(Moves, Stacks, P1, rev),
    writeans(P1),
    crane(Moves, Stacks, P2, norev),
    writeans(P2).


writeans(Stacks) :-
    maplist([[C | _], C]>>true, Stacks, Codes),
    format('~s~n', [Codes]).

crane([], S, S, _).
crane([[A, B, C] | Tail], IS, OS, R) :-
    % Maybe it's time to learn about the dictionary datastructure.
    % Manipulating lists like this is tedious.
    nth1(B, IS, FromStack, ISWOFrom),
    length(FromLift, A),
    append(FromLift, FromRem, FromStack),
    nth1(B, IS1, FromRem, ISWOFrom),

    nth1(C, IS1, ToStack, ISWOTo),
    (R == rev -> reverse(FromLift, FromLiftR) ; FromLiftR = FromLift),
    append(FromLiftR, ToStack, NewToStack),
    nth1(C, IS2, NewToStack, ISWOTo),

    crane(Tail, IS2, OS, R).
