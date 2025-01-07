main :-
    read_line_to_string(user_input, S), split_string(S, ",", "", Commands),
    maplist(hash, Commands, Hashes), sum_list(Hashes, Part1),

    length(Buckets, 256), maplist(=([]), Buckets),
    foldl(apply_command, Commands, Buckets, NBuckets),
    aggregate_all(sum(B*I*L), (
        nth1(B, NBuckets, Bucket), nth1(I, Bucket, _-L)
    ), Part2),
    format("Part 1: ~d~nPart 2: ~d~n", [Part1, Part2]).

hash(S, H) :-
    string_codes(S, Codes),
    foldl([X, Y, Z]>>(Z is (X + Y) * 17 mod 256), Codes, 0, H).

apply_command(Command, Buckets, NBuckets) :-
    (string_concat(Label, "-", Command) -> L is -1;
        split_string(Command, "=", "", [Label, LS]),
        number_string(L, LS)),
    hash(Label, B),
    nth0(B, Buckets, Bucket, Rest),
    (L =:= -1 ->
        (selectchk(Label-_, Bucket, NBucket) -> true; NBucket = Bucket);
        (selectchk(Label-_, Bucket, Label-L, NBucket) -> true;
            append(Bucket, [Label-L], NBucket))),
    nth0(B, NBuckets, NBucket, Rest).
