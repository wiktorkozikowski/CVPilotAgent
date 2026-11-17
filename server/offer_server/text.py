"""Text helpers shared by the importer and the ranking."""

import re
import unicodedata

# Offers spell the same city in several ways ("Kraków", "Krakow", "Warsaw").
CITY_ALIASES = {"warsaw": "warszawa", "cracow": "krakow"}


def plain(text: str) -> str:
    """Lower case without Polish diacritics, so "Kraków" equals "Krakow"."""
    text = text.casefold().replace("ł", "l")
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def city_key(text: str) -> str:
    key = plain(text).strip()
    return CITY_ALIASES.get(key, key)


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", plain(text)))


def contains_term(text: str, term: str) -> bool:
    """True when `term` occurs in `text` as a whole word or phrase ("C" not in "CSS").

    Both sides are compared without case and diacritics. Symbols such as "+" or
    "#" stay part of the term, so "C++" and "C#" are told apart.
    """
    needle = plain(term).strip()
    if not needle:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9+#])"
    return re.search(pattern, plain(text)) is not None
