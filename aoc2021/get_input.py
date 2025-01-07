#!/usr/bin/env python3

"""
Fetch AoC input as files.
By Asger Hautop Drewsen: https://github.com/Tyilo
"""

import sys
from pathlib import Path
from typing import Optional

import requests

YEAR = 2021
URL_PREFIX = f"https://adventofcode.com/{YEAR}"


def validate_session(session):
    test_url = f"{URL_PREFIX}/settings"
    r = session.get(test_url)
    return r.status_code == 200 and r.url == test_url


session_cookie: Optional[str]
try:
    with open(".session", "r") as f:
        session_cookie = f.read().strip()
except FileNotFoundError:
    session_cookie = None

while True:
    if not session_cookie:
        session_cookie = input("Session cookie value: ").strip()
        with open(".session", "w") as f:
            f.write(session_cookie)

    session = requests.Session()
    session.cookies.set("session", session_cookie, domain=".adventofcode.com", path="/")
    if validate_session(session):
        break

    print("That session cookie doesn't seem to work. Try again.")
    session_cookie = None


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
