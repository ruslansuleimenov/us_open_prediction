"""Parsing the official US Open draw out of its PDF."""

import re
from pathlib import Path

import pandas as pd

# [seed] (entry) slot. SURNAME, Firstname COUNTRY
# The country is optional: neutral athletes are printed without a flag.
LINE = re.compile(
    r"^(?:\[(?P<seed>\d+)\])?"
    r"(?:\((?P<entry>[QWL])\))?"
    r"(?P<slot>\d+)\.\s*"
    r"(?P<surname>[A-Z][A-Z'’\- ]*[A-Z]),\s*"
    r"(?P<first>.+?)"
    r"(?:\s+(?P<country>[A-Z]{3}))?$"
)


def parse_draw(pdf_path: str | Path) -> pd.DataFrame:
    """Draw PDF -> a table of 128 rows, one per slot."""
    from pypdf import PdfReader

    rows = []
    for page in PdfReader(str(pdf_path)).pages:
        for line in page.extract_text().splitlines():
            m = LINE.match(line.strip())
            if m:
                rows.append(m.groupdict())

    draw = pd.DataFrame(rows)
    draw["slot"] = draw["slot"].astype(int)
    draw["seed"] = pd.to_numeric(draw["seed"])
    draw = draw.sort_values("slot").reset_index(drop=True)
    draw["draw_name"] = draw["surname"] + ", " + draw["first"]
    return draw


def to_dataset_name(surname: str, first: str) -> str:
    """"ZVEREV, Alexander" -> "Zverev A.", the name format used in our data.

    The surname is title-cased and the given names are reduced to initials, so
    compound names such as "Juan Manuel" become "J.M." as the dataset has them.
    """
    sur = " ".join(w.capitalize() for w in surname.split())
    sur = "-".join(w[:1].upper() + w[1:] for w in sur.split("-"))
    initials = "".join(part[0].upper() + "." for part in re.split(r"[\s\-]+", first) if part)
    return f"{sur} {initials}"


# Players whose name is spelled differently in the draw than in the dataset.
# Each one was checked by hand against match counts and dates — see the README.
NAME_ALIASES = {
    "Burruchaga R.A.": "Burruchaga R.",
    "Vallejo A.D.": "Vallejo D.",            # dataset uses the second given name
    "Barrios Vera T.": "Barrios Vera M.T.",
    "Merida D.": "Merida Aguilar D.",        # dataset uses the double surname
    "Etcheverry T.M.": "Etcheverry T.",
    "Wolf J.": "Wolf J.J.",
}


def resolve_names(draw: pd.DataFrame, known: set[str]) -> pd.DataFrame:
    """Add `name` (the dataset spelling) and `known` (is there any history).

    Players without history are debutants from qualifying and wildcards; they
    fall back to default state — Elo 1500 and cold-start form.
    """
    names = [to_dataset_name(r.surname, r.first) for r in draw.itertuples()]
    names = [NAME_ALIASES.get(n, n) for n in names]
    out = draw.copy()
    out["name"] = names
    out["known"] = out["name"].isin(known)
    return out
