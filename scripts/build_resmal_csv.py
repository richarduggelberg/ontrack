"""Fetch "Lista over avsnitt av Pa sparet" from Wikipedia and extract all
"Resmal" (destinations) in chronological order into data/resmal.csv.

Usage: python scripts/build_resmal_csv.py
"""
import csv
import html
import re
import urllib.request
from pathlib import Path

URL = (
    "https://sv.wikipedia.org/w/index.php?"
    "title=Lista_%C3%B6ver_avsnitt_av_P%C3%A5_sp%C3%A5ret&action=raw"
)

MONTHS = {
    "januari": 1, "februari": 2, "mars": 3, "april": 4, "maj": 5, "juni": 6,
    "juli": 7, "augusti": 8, "september": 9, "oktober": 10,
    "november": 11, "december": 12,
}

SEASON_RE = re.compile(r"^==\s*S[äa]song\s+(\d+)\s*(?:\(([^)]*)\))?\s*==", re.MULTILINE)
DATE_RE = re.compile(r"(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(\d{4})")


def strip_wiki_markup(text: str) -> str:
    """Strip refs/templates/small-tags and return the human-readable display text."""
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"</?small>", "", text)
    text = html.unescape(text).replace("\xa0", " ")
    return text.strip()


def wikilink_display(link: str) -> str:
    """[[Target|Display]] -> Display, [[Target]] -> Target."""
    inner = link[2:-2]
    if "|" in inner:
        inner = inner.rsplit("|", 1)[-1]
    return inner.strip()


def extract_destinations(cell: str) -> list[str]:
    cell = strip_wiki_markup(cell)
    parts = re.split(r"<br\s*/?>", cell)
    dests = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        links = re.findall(r"\[\[[^\[\]]*\]\]", part)
        if links:
            for link in links:
                name = wikilink_display(link)
                if name:
                    dests.append(name)
        else:
            plain = part.strip(" ?\u2013-")
            if plain:
                dests.append(plain)
    return dests


def parse_date(raw: str):
    raw = strip_wiki_markup(raw)
    m = DATE_RE.search(raw)
    if not m:
        return ""
    day, month_name, year = m.groups()
    return f"{year}-{MONTHS[month_name]:02d}-{int(day):02d}"


def split_row_cells(block: str) -> list[str]:
    """Split one '|-' separated row block into its cell contents."""
    cells = []
    current = None
    for line in block.splitlines():
        if not line.strip():
            continue
        if line.startswith("!"):
            line = "|" + line[1:]
        if line.startswith("|"):
            if current is not None:
                cells.append(current.strip())
            content = line[1:]
            while content.startswith("|"):
                content = content[1:]
            sub_cells = re.split(r"\|\|", content)
            current = sub_cells[0]
            for extra in sub_cells[1:]:
                cells.append(current.strip())
                current = extra
        elif current is not None:
            current += "\n" + line
    if current is not None:
        cells.append(current.strip())
    return cells


def cell_text(cell: str) -> str:
    """Drop a leading 'attr | value' style prefix (e.g. colspan=2 |)."""
    if "|" in cell and "[[" not in cell.split("|", 1)[0]:
        prefix, rest = cell.split("|", 1)
        if "=" in prefix or prefix.strip() == "":
            return rest.strip()
    return cell.strip()


def parse_header_cells(block: str) -> list[str]:
    """Parse a header row, expanding 'colspan=N | text' into N slots so
    the resulting list lines up positionally with the data rows (which
    have one cell per actual column, not per header)."""
    expanded = []
    for raw in split_row_cells(block):
        colspan = 1
        text = raw
        m = re.match(r"\s*colspan\s*=\s*\"?(\d+)\"?\s*\|\s*(.*)", raw, re.DOTALL)
        if m:
            colspan = int(m.group(1))
            text = m.group(2)
        else:
            text = cell_text(raw)
        text = strip_wiki_markup(text)
        expanded.extend([text] * colspan)
    return expanded


def parse_table(table_text: str):
    rows_raw = table_text.split("|-")
    headers = []
    data_rows = []
    for i, block in enumerate(rows_raw):
        if i == 0:
            headers = parse_header_cells(block)
            continue
        cells = [cell_text(c) for c in split_row_cells(block)]
        if cells:
            data_rows.append(cells)
    return headers, data_rows


def find_table(season_text: str):
    marker = "Program och resultat"
    idx = season_text.find(marker)
    if idx == -1:
        return None
    start = season_text.find("{|", idx)
    if start == -1:
        return None
    end = season_text.find("\n|}", start)
    if end == -1:
        return None
    return season_text[start:end]


def main():
    print(f"Fetching {URL} ...")
    req = urllib.request.Request(URL, headers={"User-Agent": "ontrack-resmal-script/1.0"})
    with urllib.request.urlopen(req) as resp:
        wikitext = resp.read().decode("utf-8")

    season_starts = list(SEASON_RE.finditer(wikitext))
    out_rows = []

    for i, m in enumerate(season_starts):
        season_num = int(m.group(1))
        season_label = (m.group(2) or "").strip()
        start = m.end()
        end = season_starts[i + 1].start() if i + 1 < len(season_starts) else len(wikitext)
        season_text = wikitext[start:end]

        table_text = find_table(season_text)
        if not table_text:
            continue
        headers, data_rows = parse_table(table_text)

        lower_headers = [h.lower() for h in headers]

        def col_index(name: str):
            for idx, h in enumerate(lower_headers):
                if name in h:
                    return idx
            return None

        idx_avsnitt = col_index("avsnitt")
        idx_datum = col_index("datum")
        idx_resmal = col_index("resmål")
        if idx_resmal is None:
            idx_resmal = col_index("resmal")
        idx_matchtyp = col_index("matchtyp")
        # Other segments that also reveal a place: "Tintin & Haddock" (sasong 8-9)
        # and "Närmast vinner" (fr.o.m. sasong 32), blind-map bonus round.
        idx_tintin = col_index("tintin")
        idx_narmast = col_index("närmast vinner")
        if idx_narmast is None:
            idx_narmast = col_index("narmast vinner")

        if idx_resmal is None:
            continue

        for row in data_rows:
            if len(row) <= idx_resmal:
                continue
            avsnitt = strip_wiki_markup(row[idx_avsnitt]) if idx_avsnitt is not None and idx_avsnitt < len(row) else ""
            datum_raw = row[idx_datum] if idx_datum is not None and idx_datum < len(row) else ""
            datum = parse_date(datum_raw)
            matchtyp = strip_wiki_markup(row[idx_matchtyp]) if idx_matchtyp is not None and idx_matchtyp < len(row) else ""

            def add_rows(dests, typ):
                for order, dest in enumerate(dests, start=1):
                    out_rows.append({
                        "sasong": season_num,
                        "sasong_ar": season_label,
                        "avsnitt": avsnitt,
                        "datum": datum,
                        "matchtyp": matchtyp,
                        "typ": typ,
                        "ordning": order,
                        "resmal": dest,
                    })

            add_rows(extract_destinations(row[idx_resmal]), "resa")

            if idx_tintin is not None and idx_tintin < len(row):
                add_rows(extract_destinations(row[idx_tintin]), "tintin_haddock")

            if idx_narmast is not None and idx_narmast < len(row):
                add_rows(extract_destinations(row[idx_narmast]), "narmast_vinner")

    out_path = Path(__file__).resolve().parent.parent / "data" / "resmal.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sasong", "sasong_ar", "avsnitt", "datum", "matchtyp", "typ", "ordning", "resmal",
        ])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
