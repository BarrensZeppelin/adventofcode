main :-
    read_input(L),
    max_list(L, Best),
    writeln(Best),
    msort(L, Sorted),
    reverse(Sorted, [A, B, C | _]),
    P2 is A + B + C,
    writeln(P2).

read_elf("", 0).
read_elf(end_of_file, 0).
read_elf(Line, E) :-
    atom_number(Line, X),
    read_line_to_string(user_input, NLine),
    read_elf(NLine, Y),
    E is X + Y.

read_input(L) :-
    read_line_to_string(user_input, Line),
    (Line == end_of_file, L = [];
        read_elf(Line, E),
        read_input(Tail),
        L = [E | Tail]
    ).
