"""Test NER with complete word extraction. Long text is split into sentences, NER runs per sentence, then results are combined."""

import os
import re
from collections import Counter
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Get the project root directory (parent of scripts folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "secbert_ner_final")

# Model max length (BERT limit)
MAX_LENGTH = 512


def split_into_sentences(text):
    """Split text into sentences and return list of (sentence_text, start_offset_in_original)."""
    if not text or not text.strip():
        return []
    text = text.strip()
    # Split on sentence-ending punctuation followed by whitespace or end
    pattern = r'(?<=[.!?])\s+'
    parts = re.split(pattern, text)
    result = []
    current = 0
    for p in parts:
        s = p.strip()
        if not s:
            continue
        idx = text.find(s, current)
        if idx == -1:
            idx = current
        result.append((s, idx))
        current = idx + len(s)
    return result


def _is_token_char(char):
    """Check if character is part of a cybersecurity token."""
    valid_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:/')
    return char in valid_chars


def extract_entities_from_segment(segment_text, tokenizer, model):
    """
    Run NER on a single segment (e.g. one sentence). Returns list of dicts:
    [{'word': str, 'type': str, 'start': int, 'end': int}] with start/end relative to segment_text.
    """
    # Truncate if over max length to avoid model errors
    inputs = tokenizer(
        segment_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    offset_mapping = inputs.pop('offset_mapping')[0].tolist()
    predictions = model(**inputs).logits.argmax(dim=-1)[0].tolist()

    # Extract words with character spans from segment
    words = []
    i = 0
    while i < len(segment_text):
        if _is_token_char(segment_text[i]):
            start = i
            while i < len(segment_text) and _is_token_char(segment_text[i]):
                i += 1
            words.append({
                'text': segment_text[start:i],
                'start': start,
                'end': i,
                'labels': []
            })
        else:
            i += 1

    for idx in range(1, len(predictions) - 1):
        pred_id = predictions[idx]
        label = model.config.id2label[pred_id]
        char_start, char_end = offset_mapping[idx]
        if char_start == char_end:
            continue
        token_mid = (char_start + char_end) // 2
        for word in words:
            if word['start'] <= token_mid < word['end']:
                word['labels'].append(label)
                break

    entities = []
    for word in words:
        if not word['labels']:
            continue
        non_o_labels = [l for l in word['labels'] if l != 'O']
        if not non_o_labels:
            continue
        entity_types = [lbl.split('-')[-1] if '-' in lbl else lbl for lbl in non_o_labels]
        dominant_label = Counter(entity_types).most_common(1)[0][0]
        entities.append({
            'word': word['text'],
            'type': dominant_label,
            'start': word['start'],
            'end': word['end'],
        })
    return entities


def extract_entities(text, tokenizer, model):
    """
    Split long text into sentences, run NER on each sentence, combine results.
    Returns: (per_sentence_results, combined_entities)
    - per_sentence_results: list of {'sentence': str, 'start_offset': int, 'entities': [...]}
    - combined_entities: list of {'word': str, 'type': str, 'start': int, 'end': int} with global offsets
    """
    sentences_with_offsets = split_into_sentences(text)
    if not sentences_with_offsets:
        # No sentence boundaries: treat whole text as one segment
        sentences_with_offsets = [(text.strip(), 0)] if text.strip() else []

    per_sentence = []
    combined = []
    seen_spans = set()  # global (start, end) so each span appears only once

    for sent_text, sent_start in sentences_with_offsets:
        seg_entities = extract_entities_from_segment(sent_text, tokenizer, model)
        seg_unique = []
        for e in seg_entities:
            g_start = sent_start + e['start']
            g_end = sent_start + e['end']
            key = (g_start, g_end)
            if key in seen_spans:
                continue
            seen_spans.add(key)
            seg_unique.append(e)
        for e in seg_unique:
            combined.append({
                'word': e['word'],
                'type': e['type'],
                'start': sent_start + e['start'],
                'end': sent_start + e['end'],
            })
        per_sentence.append({
            'sentence': sent_text,
            'start_offset': sent_start,
            'entities': [{'word': e['word'], 'type': e['type']} for e in seg_unique],
        })

    return per_sentence, combined

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
print("Model loaded!\n")

print("="*70)
print("Interactive Cybersecurity NER System")
print("="*70)
print("Long text is split into sentences; NER runs per sentence, then results are combined.")
print("Entity types: MALWARE, APT, TOOL, VULNERABILITY, CVE, etc.")
print("\nCommands:")
print("  quit, exit, q - Stop the program")
print("="*70)
print()

# Interactive loop
while True:
    try:
        user_input = input("Enter your question or text: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q', '']:
            print("\nThank you for using the NER system. Goodbye!")
            break

        per_sentence, combined = extract_entities(user_input, tokenizer, model)

        print(f"\nInput ({len(per_sentence)} sentence(s)):")
        print("-" * 50)
        for i, item in enumerate(per_sentence, 1):
            print(f"  [{i}] {item['sentence'][:80]}{'...' if len(item['sentence']) > 80 else ''}")
        print()

        if len(per_sentence) > 1:
            print("ENTITIES PER SENTENCE:")
            for i, item in enumerate(per_sentence, 1):
                if item['entities']:
                    print(f"  Sentence {i}:")
                    for e in item['entities']:
                        print(f"    - {e['word']:30s} → {e['type']}")
                else:
                    print(f"  Sentence {i}: (none)")
            print()

        print("COMBINED ENTITIES (all sentences):")
        if combined:
            for e in combined:
                print(f"  {e['word']:30s} → {e['type']}")
        else:
            print("  No entities detected")
        print()

    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")
        print("Please try again.\n")
