:- dynamic(size/2), dynamic(isdir/1).

main :-
    % Very useful for debugging!
    %leash(-all), trace,
    read_string(user_input, _, Data),
    split_string(Data, "$", " \n", [_ | Commands]),
    process([], Commands), !, % Prevent backtracking over input parsing (I think)
    % Find all the directories with size less than 100000
    findall(S, (isdir(Dir), size(Dir, S), S =< 100000), Sizes),
    sum_list(Sizes, P1),
    writeln(P1),
    size(["/"], Used), % Get the total amount of used space into Used
    % Find all directories matching P2 criteria, sorted due to setof
    setof(S, Dir^(isdir(Dir), size(Dir, S), 70000000 - Used + S >= 30000000), [P2 | _]),
    writeln(P2).


isdir(["/"]).

process(_, []).
process(Pwd, [Command | Tail]) :-
    % First split a command into lines, then split the first line on spaces.
    % "ls" becomes ["ls"], "cd X" becomes ["cd", X].
    split_string(Command, "\n", "", [Cmd | CTail]),
    split_string(Cmd, " ", "", CSplit),
    % Process this command, then proceed with the remaining commands.
    process([CSplit | CTail], Pwd, NPwd),
    process(NPwd, Tail).

process([["cd", "/"]] , _, ["/"]).
process([["cd", ".."]], [_ | Pwd], Pwd).
process([["cd", Dir]] , Pwd, [Dir | Pwd]).
process([["ls"] | L], Pwd, Pwd) :-
    maplist([S, O]>>split_string(S, " ", "", O), L, LS),
    maplist({Pwd}/[[DirOrSize, Name]]>>(
        NP = [Name | Pwd],
        % The current directory contains the child.
        assertz(contains(Pwd, NP)),
        (DirOrSize == "dir" ->
            % If the listing is a directory, then assert it.
            assertz(isdir(NP));
            % Otherwise assert the size of the file.
            number_string(S, DirOrSize),
            assertz(size(NP, S)))
    ), LS).

size(Path, Size) :-
    % We only want to compute sizes for directories.
    % File sizes are asserted into the fact database.
    isdir(Path),
    % Get a list of the sizes of entries that the directory contains.
    findall(S, (contains(Path, NP), size(NP, S)), Sizes),
    sum_list(Sizes, Size).
