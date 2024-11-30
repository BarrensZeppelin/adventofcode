#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import requests

YEAR = 2024
URL_PREFIX = f"https://adventofcode.com/{YEAR}"


def validate_session(session: requests.Session) -> bool:
    test_url = f"{URL_PREFIX}/settings"
    r = session.get(test_url)
    return r.status_code == 200 and r.url == test_url


def get_session() -> requests.Session:
    session_cookie: str | None
    try:
        with open(".session") as f:
            session_cookie = f.read().strip()
    except FileNotFoundError:
        session_cookie = None

    while True:
        if not session_cookie:
            session_cookie = input("Session cookie value: ").strip()
            with open(".session", "w") as f:
                f.write(session_cookie)

        session = requests.Session()
        session.headers["User-Agent"] = "get_input.py @ github.com/BarrensZeppelin/adventofcode2024"
        session.cookies.set("session", session_cookie, domain=".adventofcode.com", path="/")
        if validate_session(session):
            return session

        print("That session cookie doesn't seem to work. Try again.")
        session_cookie = None


def submit_answer(day: int, part: Literal[1, 2], answer: str) -> str:
    session = get_session()
    r = session.post(f"{URL_PREFIX}/day/{day}/answer", data={"level": part, "answer": answer})
    r.raise_for_status()
    return r.text


if __name__ == "__main__":
    """
    Fetch AoC input as files.
    By Asger Hautop Drewsen: https://github.com/Tyilo
    """
    session = get_session()

    for i in range(1, 26):
        path = Path(f"{i:02}.in")

        if path.exists():
            continue

        r = session.get(f"{URL_PREFIX}/day/{i}/input")
        if r.ok:
            with path.open("wb") as fb:
                fb.write(r.content)
            print(f"Downloaded {path.name}")
        else:
            if r.status_code == 404:
                print(f"Day {i} not released yet")
                break
            else:
                sys.exit(f"Got unknown status code: {r.status_code}\n{r.text.strip()}")
