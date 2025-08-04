import re
from .helpers import dateparser_detect, describe_relative_date, name_list_detect
from .ner_redact import process_long_text, validate_offsets, get_entity_placeholder


def detect_and_redact_phi(text_input, ner_pipeline=None):
    # Step 1: Kenyan name detection
    names_found = name_list_detect(text_input)
    text_redacted = text_input

    # Step 2: Date detection and replacement
    dates_found = dateparser_detect(text_input)
    for match, dt in dates_found:
        relative = describe_relative_date(dt)
        text_redacted = text_redacted.replace(match, relative)

    # Step 3: Kenyan name redaction
    for name in names_found:
        # Use regex with word boundaries for precise matching
        pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
        text_redacted = pattern.sub("[name]", text_redacted)

    # Step 4: NER-based redaction (if pipeline available)
    ner_entities = []
    if ner_pipeline:
        # Process text through NER pipeline
        ents = process_long_text(text_redacted, ner_pipeline)
        validated_ents = validate_offsets(text_redacted, ents)

        # Create sorted replacement list
        replacements = []
        for ent in validated_ents:
            if ent["entity_group"] in {"PER", "LOC", "ORG"}:
                placeholder = get_entity_placeholder(ent["entity_group"])
                replacements.append((ent["start"], ent["end"], placeholder))
                ner_entities.append(ent["word"])

        # Apply replacements in reverse order to avoid offset issues
        replacements.sort(key=lambda x: x[0], reverse=True)
        for start, end, placeholder in replacements:
            text_redacted = text_redacted[:start] + placeholder + text_redacted[end:]

    # Determine if any PHI was detected
    phi_detected = bool(names_found or dates_found or ner_entities)

    return {
        "phi_detected": phi_detected,
        "kenyan_name_matches": names_found,
        "dates": [d[0] for d in dates_found],
        "ner_entities": ner_entities,
        "redacted_text": text_redacted,
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
