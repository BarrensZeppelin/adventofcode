main :-
    % Very useful for debugging!
    %leash(-all), trace,
    read_line_to_string(user_input, Inp),
    process(Inp, 4),
    process(Inp, 14).

process(Inp, N) :-
    aggregate_all(min(I + N), (sub_string(Inp, I, N, _, Sub), string_chars(Sub, Codes), is_set(Codes)), Ans),
    writeln(Ans).

