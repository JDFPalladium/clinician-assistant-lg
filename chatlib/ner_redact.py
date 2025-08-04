import os
import ssl
import certifi
import sys
import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.pipelines import pipeline


if sys.platform in ["darwin", "win32"]:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

try:
    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where()
    )  # pylint: disable=protected-access
except AttributeError:
    pass

MODEL_NAME = "Davlan/xlm-roberta-large-masakhaner"
LOCAL_MODEL_PATH = "./local_model"


def load_ner_model():
    """Robust cross-platform model loader with offline support"""
    os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_PATH, use_fast=True, local_files_only=True
        )
    except (OSError, ValueError):
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        except ImportError:

            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        tokenizer.save_pretrained(LOCAL_MODEL_PATH)

    try:
        model = AutoModelForTokenClassification.from_pretrained(
            LOCAL_MODEL_PATH, local_files_only=True
        )

    except (OSError, ValueError):
        model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
        model.save_pretrained(LOCAL_MODEL_PATH)

    device = 0 if torch.cuda.is_available() else -1
    if sys.platform == "darwin":
        device = "mps" if torch.backends.mps.is_available() else -1

    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="max",
        device=device,
    )


def validate_offsets(text: str, entities: list) -> list:
    """Robust entity validation with normalized comparison"""
    valid_entities = []
    for ent in entities:
        start, end = ent["start"], ent["end"]

        if not (0 <= start < end <= len(text)):
            continue

        extracted = text[start:end]
        normalized_extracted = re.sub(r"\W+", "", extracted).lower()
        normalized_word = re.sub(r"\W+", "", ent["word"]).lower()

        if normalized_extracted == normalized_word:
            valid_entities.append(ent)
        else:
            print(f"Offset mismatch: '{ent['word']}' vs actual '{extracted}'")
    return valid_entities


def redact_text(text: str, entities: list, placeholder: str = "[REDACTED]") -> str:
    """Efficient redaction with sorted offsets"""

    target_entities = [
        e for e in entities if e["entity_group"] in {"PER", "LOC", "ORG"}
    ]
    sorted_entities = sorted(target_entities, key=lambda x: x["start"], reverse=True)

    segments = []
    last_end = len(text)

    for ent in sorted_entities:
        start, end = ent["start"], ent["end"]
        if end > last_end:
            continue

        segments.append(text[end:last_end])
        segments.append(placeholder)
        last_end = start

    segments.append(text[:last_end])
    return "".join(reversed(segments))


def process_long_text(text: str, ner_pipeline, max_tokens: int = 500) -> list:
    """Process long text with sliding window approach"""
    if len(text) <= max_tokens * 4:
        return ner_pipeline(text)

    sentence_endings = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s"
    sentences = re.split(sentence_endings, text)

    entities = []
    chunk = ""
    chunk_start = 0
    window_size = 3

    for i in range(0, len(sentences), window_size):
        chunk = " ".join(sentences[i : i + window_size])
        ents = ner_pipeline(chunk)

        for ent in ents:
            ent["start"] += chunk_start
            ent["end"] += chunk_start

        entities.extend(ents)
        chunk_start += len(chunk) + 1

    return entities


def get_entity_placeholder(entity_group: str) -> str:
    """Return specific placeholder for entity type"""
    placeholders = {
        "PER": "[name]",
        "LOC": "[location]",
        "ORG": "[organization]"
    }
    return placeholders.get(entity_group, "[redacted]")

