"""
ANSI color helpers for CLI output, shared by chatbot.py and the pipeline
answer() functions so questions/status/answers are visually distinct in a
terminal. Uses colorama so it also works on older Windows consoles that
don't do ANSI natively.
"""
from __future__ import annotations

import colorama

colorama.init(autoreset=True)


def question(text: str) -> str:
    return f"{colorama.Fore.CYAN}{text}{colorama.Style.RESET_ALL}"


def answer(text: str) -> str:
    return f"{colorama.Fore.GREEN}{text}{colorama.Style.RESET_ALL}"


def status(text: str) -> str:
    return f"{colorama.Style.DIM}{text}{colorama.Style.RESET_ALL}"


def warn(text: str) -> str:
    return f"{colorama.Fore.YELLOW}{text}{colorama.Style.RESET_ALL}"
