from pathlib import Path
import re
import dateparser.search
from datetime import datetime
from dateutil.relativedelta import relativedelta

RELATIVE_INDICATORS = [
    "ago",
    "later",
    "before",
    "after",
    "yesterday",
    "tomorrow",
    "today",
    "tonight",
    "last",
    "next",
    "this",
    "coming",
    "previous",
    "past",
]


def is_relative_date(text_relative):
    text_lower = text_relative.lower()
    return any(word in text_lower for word in RELATIVE_INDICATORS)


def load_kenyan_names(filepath="data/kenyan_names.txt"):
    if not Path(filepath).exists():
        return set()
    with open(filepath, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())


kenyan_names = load_kenyan_names()


def name_list_detect(text_names):
    words = re.findall(r"\b\w+\b", text_names)
    matches = [w for w in words if w.lower() in kenyan_names]
    return matches


def dateparser_detect(text_dates):
    results_date = dateparser.search.search_dates(text_dates, languages=["en"])
    if not results_date:
        return []
    filtered = [r for r in results_date if not is_relative_date(r[0])]
    return filtered


def describe_relative_date(dt, reference=None):
    if reference is None:
        reference = datetime.now()

    delta = relativedelta(reference, dt)

    if delta.years > 0:
        return f"{delta.years} year{'s' if delta.years > 1 else ''} ago"
    elif delta.months > 0:
        return f"{delta.months} month{'s' if delta.months > 1 else ''} ago"
    elif delta.days >= 7:
        weeks = delta.days // 7
        return f"{weeks} week{'s' if weeks > 1 else ''} ago"
    elif delta.days > 0:
        return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
    else:
        return "today"


def detect_and_redact_phi(text_input):
    names_found = name_list_detect(text_input)
    dates_found = dateparser_detect(text_input)

    phi_detected = bool(names_found or dates_found)

    for match, dt in dates_found:
        relative = describe_relative_date(dt)
        text_input = text_input.replace(match, relative)

    for name in names_found:
        pattern = re.compile(rf"\b{name}\b", re.IGNORECASE)
        text_input = pattern.sub("[name]", text_input)

    return {
        "phi_detected": phi_detected,
        "kenyan_name_matches": names_found,
        "dates": [d[0] for d in dates_found],
        "redacted_text": text_input,
    }


if __name__ == "__main__":
    print("\n🔍 PHI Detection Tool (Kenyan context + redaction with relative dates)\n")
    while True:
        text = input("Enter clinical text (or 'q' to quit):\n> ")
        if text.lower() == "q":
            break
        results = detect_and_redact_phi(text)

        if results["phi_detected"]:
            print("\n⚠️  Possible PHI detected!")
            if results["kenyan_name_matches"]:
                print(" - Possible Kenyan names:", results["kenyan_name_matches"])
            if results["dates"]:
                print(" - Dates detected:", results["dates"])

            print("\n🛡️  Redacted text:")
            print(results["redacted_text"])
        else:
            print("\n✅ No PHI detected.")
        print("\n---\n")
