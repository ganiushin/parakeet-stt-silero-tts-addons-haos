"""Text normalization for the Russian Silero v5 model.

The model silently drops anything it has no symbol for — digits, Latin
script, emoji — so "Сейчас 13:45" would be spoken as "Сейчас". Numbers are
expanded to Russian words with num2words and Latin words are (optionally)
transliterated to Cyrillic.

Expansion also has to agree grammatically with what follows, because Assist
reads out measurements constantly: "21,5°C" is "двадцать одна целая пять
десятых градуса", not "двадцать один и пять градусов".
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from num2words import num2words

_CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")
_LATIN = re.compile(r"[a-zA-Z]+")
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_MINUS = re.compile(r"(?:^|(?<=[\s(]))[-−](?=\d)")
_WORD_AFTER = re.compile(r"\s*([а-яё]+)", re.IGNORECASE)
# Anything the model has no symbol for becomes a space (after digits and
# Latin have already been rewritten).
_UNSPEAKABLE = re.compile(r"[^а-яёА-ЯЁ\s.,!?…:;()\-—«»\"']")
_SPACES = re.compile(r"\s+")

# Unit symbols that follow a number, with the three count forms Russian needs
# ("1 градус", "2 градуса", "5 градусов") and any invariant tail.
_SUFFIX_UNITS = {
    "°C": (("градус", "градуса", "градусов"), " Цельсия"),
    "°F": (("градус", "градуса", "градусов"), " Фаренгейта"),
    "°": (("градус", "градуса", "градусов"), ""),
    "%": (("процент", "процента", "процентов"), ""),
}
_QUANTITY = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(" + "|".join(re.escape(s) for s in _SUFFIX_UNITS) + r")"
)
# Same symbols standing on their own, plus the prefix ones.
_LONE_UNITS = [("°C", " градусов Цельсия "), ("°F", " градусов Фаренгейта "),
               ("°", " градусов "), ("%", " процентов "), ("№", " номер ")]

# A numeral only reveals gender at 1 and 2, and the feminine nouns Assist
# actually puts after a bare number are durations and "тысяча". Anything else
# takes the masculine form, which is also how a lone number should sound.
_FEMININE_NOUNS = {
    "минута", "минуты", "минут", "минуту",
    "секунда", "секунды", "секунд", "секунду",
    "тысяча", "тысячи", "тысяч", "тысячу",
    "неделя", "недели", "недель", "неделю",
}

# Rough Latin-to-Cyrillic transliteration: digraphs first, then single
# letters. Not linguistically perfect — the goal is that "Spotify" is spoken
# recognizably instead of dropped.
_DIGRAPHS = [
    ("shch", "щ"), ("sch", "щ"), ("sh", "ш"), ("ch", "ч"), ("zh", "ж"),
    ("kh", "х"), ("ts", "ц"), ("yo", "ё"), ("yu", "ю"), ("ya", "я"),
    ("ye", "е"), ("ck", "к"), ("th", "т"), ("ph", "ф"), ("oo", "у"),
    ("ee", "и"),
]
_LETTERS = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г",
    "h": "х", "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в", "w": "в", "x": "кс", "y": "и", "z": "з",
}


def _num(n: int, gender: str = "m") -> str:
    return num2words(n, lang="ru", gender=gender)


def _count_form(n: int, forms: tuple[str, str, str]) -> str:
    """Pick the Russian count form: 1 градус, 2-4 градуса, 5-20 градусов."""
    n = abs(n) % 100
    if 11 <= n <= 14:
        return forms[2]
    n %= 10
    if n == 1:
        return forms[0]
    if 2 <= n <= 4:
        return forms[1]
    return forms[2]


def _spell_number(literal: str, gender: str = "m") -> str:
    """Spell a written number, keeping fractions as proper Russian fractions.

    Decimal (not float) so that "21,50" stays "пятьдесят сотых" and binary
    rounding can never leak into speech.
    """
    if "." in literal or "," in literal:
        try:
            return num2words(Decimal(literal.replace(",", ".")), lang="ru")
        except InvalidOperation:
            return literal
    return _num(int(literal), gender)


def _is_fraction(literal: str) -> bool:
    return "." in literal or "," in literal


def _expand_time(m: re.Match) -> str:
    hours, minutes = int(m.group(1)), int(m.group(2))
    if hours > 23 or minutes > 59:  # a score like "30:15", not a time
        return f"{_num(hours)} {_num(minutes)}"
    if minutes == 0:
        return f"{_num(hours)} ноль ноль"
    if minutes < 10:
        return f"{_num(hours)} ноль {_num(minutes)}"
    return f"{_num(hours)} {_num(minutes)}"


def _expand_quantity(m: re.Match) -> str:
    """Expand "21,5°C" with the unit agreeing with the number."""
    literal, symbol = m.group(1), m.group(2)
    forms, tail = _SUFFIX_UNITS[symbol]
    if _is_fraction(literal):
        # A fractional quantity takes the genitive singular:
        # "двадцать одна целая пять десятых градуса".
        unit = forms[1]
    else:
        unit = _count_form(int(literal), forms)
    return f" {_spell_number(literal)} {unit}{tail} "


def _expand_number(m: re.Match) -> str:
    """Expand a bare number, agreeing in gender with the noun after it."""
    literal = m.group(0)
    gender = "m"
    if not _is_fraction(literal):
        following = _WORD_AFTER.match(m.string, m.end())
        if following and following.group(1).lower() in _FEMININE_NOUNS:
            gender = "f"
    return f" {_spell_number(literal, gender)} "


def _translit_word(m: re.Match) -> str:
    word = m.group(0).lower()
    for latin, cyr in _DIGRAPHS:
        word = word.replace(latin, cyr)
    return "".join(_LETTERS.get(c, c) for c in word)


def normalize(text: str, transliterate: bool = True) -> str:
    """Return speakable text for Silero, or "" if nothing would be voiced."""
    text = _MINUS.sub("минус ", text)
    text = _TIME.sub(_expand_time, text)
    text = _QUANTITY.sub(_expand_quantity, text)
    for symbol, replacement in _LONE_UNITS:
        text = text.replace(symbol, replacement)
    text = _NUMBER.sub(_expand_number, text)
    if transliterate:
        text = _LATIN.sub(_translit_word, text)
    text = _UNSPEAKABLE.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    if not _CYRILLIC.search(text):
        return ""
    return text
