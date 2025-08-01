from pathlib import Path
import re
from .helpers import dateparser_detect, describe_relative_date


def load_kenyan_names(filepath="data/processed/kenyan_names.txt"):
    if not Path(filepath).exists():
        return set()
    with open(filepath, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())


kenyan_names = load_kenyan_names()


def name_list_detect(text_names):
    words = re.findall(r"\b\w+\b", text_names)
    matches = [w for w in words if w.lower() in kenyan_names]
    return matches


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
