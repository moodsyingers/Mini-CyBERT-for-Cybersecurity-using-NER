"""
Flask Backend API for the fine-tuned SecBERT NER model.
Provides cybersecurity entity recognition.
Long text is split into sentences; NER runs per sentence (sliding window if >512 tokens), then results are combined.
"""

import re
import json
from pathlib import Path
from collections import Counter

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Model paths
BASE_DIR = Path(__file__).parent.parent
NER_MODEL_PATH = BASE_DIR / "models" / "secbert_ner_final"
VOCABULARY_PATH = BASE_DIR / "datasets" / "cyber" / "entity_vocabulary.json"

# Model max length (BERT limit)
NER_MAX_LENGTH = 512
# Sliding window stride when a segment exceeds NER_MAX_LENGTH tokens
NER_WINDOW_STRIDE = 256

# Global model variables
ner_pipeline = None
ner_tokenizer = None
ner_model = None
ner_loaded = False

# Dataset vocabulary (loaded from cyberner_clean.csv)
dataset_vocabulary = None
vocab_loaded = False


def split_into_sentences(text):
    """Split text into sentences; return list of (sentence_text, start_offset_in_original)."""
    if not text or not text.strip():
        return []
    text = text.strip()
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
    valid_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:/')
    return char in valid_chars


def _build_words(segment_text):
    """Build list of words with character spans; each word has 'labels': []."""
    words = []
    i = 0
    while i < len(segment_text):
        if _is_token_char(segment_text[i]):
            start = i
            while i < len(segment_text) and _is_token_char(segment_text[i]):
                i += 1
            words.append({'text': segment_text[start:i], 'start': start, 'end': i, 'labels': []})
        else:
            i += 1
    return words


def _words_to_entities(words):
    """Convert words (with labels filled) to entity list; one unique entity per word (dominant type)."""
    entities = []
    for word in words:
        if not word['labels']:
            continue
        non_o_labels = [l for l in word['labels'] if l != 'O']
        if not non_o_labels:
            continue
        entity_types = [lbl.split('-')[-1] if '-' in lbl else lbl for lbl in non_o_labels]
        dominant_type = Counter(entity_types).most_common(1)[0][0]
        entities.append({
            'word': word['text'],
            'entity_type': dominant_type,
            'start': word['start'],
            'end': word['end'],
        })
    return entities


def _assign_labels_to_words(words, offset_mapping, predictions, model):
    """Assign token-level predictions to words using offset_mapping (segment-relative)."""
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


def extract_entities_from_segment_sliding_window(segment_text, tokenizer, model):
    """Run NER on a long segment using overlapping token windows. No truncation."""
    enc = tokenizer(
        segment_text,
        return_offsets_mapping=True,
        truncation=False,
        add_special_tokens=True,
    )
    input_ids_list = enc['input_ids']
    offset_mapping = enc['offset_mapping']
    n_content = len(input_ids_list) - 2

    if n_content <= 0:
        return []

    words = _build_words(segment_text)
    if not words:
        return []

    for start in range(0, n_content, NER_WINDOW_STRIDE):
        end = min(start + NER_MAX_LENGTH, n_content)
        window_ids = (
            [input_ids_list[0]]
            + input_ids_list[1 + start : 1 + end]
            + [input_ids_list[-1]]
        )
        inputs = {
            'input_ids': torch.tensor([window_ids], dtype=torch.long),
            'attention_mask': torch.ones(1, len(window_ids), dtype=torch.long),
        }
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = logits.argmax(dim=-1)[0][1:-1].tolist()

        for i in range(end - start):
            content_idx = start + i
            char_start, char_end = offset_mapping[1 + content_idx]
            if char_start == char_end:
                continue
            label = model.config.id2label[preds[i]]
            token_mid = (char_start + char_end) // 2
            for word in words:
                if word['start'] <= token_mid < word['end']:
                    word['labels'].append(label)
                    break

    return _words_to_entities(words)


def extract_entities_from_segment(segment_text, tokenizer, model):
    """Run NER on one segment. Uses sliding window if segment exceeds NER_MAX_LENGTH tokens."""
    enc = tokenizer(
        segment_text,
        return_offsets_mapping=True,
        truncation=False,
        add_special_tokens=True,
    )
    n_tokens = len(enc['input_ids'])

    if n_tokens > NER_MAX_LENGTH + 2:
        return extract_entities_from_segment_sliding_window(segment_text, tokenizer, model)

    inputs = tokenizer(
        segment_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=NER_MAX_LENGTH,
    )
    offset_mapping = inputs.pop('offset_mapping')[0].tolist()
    predictions = model(**inputs).logits.argmax(dim=-1)[0].tolist()

    words = _build_words(segment_text)
    _assign_labels_to_words(words, offset_mapping, predictions, model)
    return _words_to_entities(words)


def enhance_with_keywords(text, existing_entities):
    """
    Enhance NER results with dataset vocabulary lookup.
    This catches cybersecurity terms that the model might miss.
    Uses vocabulary extracted from cyberner_clean.csv dataset.
    """
    import re
    import sys
    
    if not vocab_loaded or dataset_vocabulary is None:
        print("[KEYWORD ENHANCE] Vocabulary not loaded, using hardcoded fallback", flush=True)
        return enhance_with_hardcoded_keywords(text, existing_entities)
    
    print(f"[DATASET ENHANCE] Input text: {text[:100]}...", flush=True)
    print(f"[DATASET ENHANCE] Existing entities: {len(existing_entities)}", flush=True)
    sys.stdout.flush()
    
    # Track existing entity spans to avoid duplicates
    existing_spans = set()
    for e in existing_entities:
        existing_spans.add((e['start'], e['end']))
    
    enhanced = list(existing_entities)
    added_count = 0
    
    # First, try multi-word entities (longer matches are more specific)
    multi_word = dataset_vocabulary.get('multi_word', {})
    for entity_phrase, entity_data in multi_word.items():
        entity_type = entity_data['entity_type']
        
        # Search for phrase (case insensitive, word boundary)
        pattern = re.compile(r'\b' + re.escape(entity_phrase) + r'\b', re.IGNORECASE)
        
        for match in pattern.finditer(text):
            start = match.start()
            end = match.end()
            
            # Check if this span overlaps with existing entities
            overlaps = False
            for e_start, e_end in existing_spans:
                if not (end <= e_start or start >= e_end):
                    overlaps = True
                    break
            
            if not overlaps:
                word = text[start:end]
                enhanced.append({
                    'word': word,
                    'entity_type': entity_type,
                    'start': start,
                    'end': end,
                    'source': 'dataset_multi_word'
                })
                existing_spans.add((start, end))
                added_count += 1
                print(f"[DATASET ENHANCE] Added multi-word: {word} -> {entity_type}", flush=True)
    
    # Then, try single words
    single_words = dataset_vocabulary.get('single_words', {})
    
    # Extract words from text using the same logic as _build_words
    words_in_text = []
    i = 0
    while i < len(text):
        if _is_token_char(text[i]):
            start = i
            while i < len(text) and _is_token_char(text[i]):
                i += 1
            word_text = text[start:i]
            words_in_text.append({
                'text': word_text,
                'start': start,
                'end': i
            })
        else:
            i += 1
    
    # Check each word against vocabulary
    for word_info in words_in_text:
        word_lower = word_info['text'].lower()
        start = word_info['start']
        end = word_info['end']
        
        # Skip if too short
        if len(word_lower) <= 1:
            continue
        
        # Check vocabulary
        if word_lower in single_words:
            entity_data = single_words[word_lower]
            entity_type = entity_data['entity_type']
            
            # Check if this span overlaps with existing entities
            overlaps = False
            for e_start, e_end in existing_spans:
                if not (end <= e_start or start >= e_end):
                    overlaps = True
                    break
            
            if not overlaps:
                enhanced.append({
                    'word': word_info['text'],
                    'entity_type': entity_type,
                    'start': start,
                    'end': end,
                    'source': 'dataset_single_word'
                })
                existing_spans.add((start, end))
                added_count += 1
                print(f"[DATASET ENHANCE] Added single word: {word_info['text']} -> {entity_type}", flush=True)
    
    print(f"[DATASET ENHANCE] Added {added_count} new entities from dataset", flush=True)
    print(f"[DATASET ENHANCE] Total entities: {len(enhanced)}", flush=True)
    sys.stdout.flush()
    return enhanced


def enhance_with_hardcoded_keywords(text, existing_entities):
    """
    Fallback: Enhance with hardcoded keywords if vocabulary file is not available.
    This is the original implementation.
    """
    import re
    import sys
    
    print(f"[KEYWORD ENHANCE] Input text: {text}", flush=True)
    print(f"[KEYWORD ENHANCE] Existing entities: {len(existing_entities)}", flush=True)
    sys.stdout.flush()
    
    # Define keyword patterns and their entity types
    keyword_patterns = {
        'MALWARE': [
            'ransomware', 'trojan', 'backdoor', 'rootkit', 'spyware', 'adware',
            'worm', 'virus', 'conti', 'ryuk', 'maze', 'wannacry', 'petya',
            'notpetya', 'cryptolocker', 'trickbot', 'emotet', 'dridex',
            'lockbit', 'blackcat', 'alphv', 'revil', 'sodinokibi', 'malware'
        ],
        'VULNERABILITY': [
            'vulnerability', 'vulnerabilities', 'exploit', 'zero-day',
            'zeroday', 'security flaw', 'security issue'
        ],
        'THREAT_ACTOR': [
            'apt28', 'apt29', 'apt32', 'apt40', 'fancy bear', 'cozy bear',
            'lazarus', 'sandworm', 'equation group', 'carbanak', 'fin7', 'fin8'
        ],
        'METHOD': [
            'phishing', 'spear-phishing', 'spear phishing', 'social engineering', 
            'brute force', 'credential stuffing', 'sql injection', 'xss', 'ddos', 'dos',
            'man-in-the-middle', 'mitm', 'privilege escalation', 'lateral movement'
        ],
        'TOOL': [
            'cobalt strike', 'metasploit', 'mimikatz', 'powershell', 'psexec',
            'nmap', 'burp suite', 'wireshark', 'hydra', 'john the ripper'
        ]
    }
    
    # Track existing entity spans to avoid duplicates
    existing_spans = set()
    for e in existing_entities:
        existing_spans.add((e['start'], e['end']))
    
    enhanced = list(existing_entities)
    added_count = 0
    
    for entity_type, keywords in keyword_patterns.items():
        for keyword in keywords:
            # Use word boundary matching for better accuracy
            # Case insensitive search
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            
            for match in pattern.finditer(text):
                start = match.start()
                end = match.end()
                
                # Check if this span overlaps with existing entities
                overlaps = False
                for e_start, e_end in existing_spans:
                    if not (end <= e_start or start >= e_end):
                        overlaps = True
                        break
                
                if not overlaps:
                    # Add the keyword-detected entity
                    word = text[start:end]
                    enhanced.append({
                        'word': word,
                        'entity_type': entity_type,
                        'start': start,
                        'end': end
                    })
                    existing_spans.add((start, end))
                    added_count += 1
                    print(f"[KEYWORD ENHANCE] Added: {word} -> {entity_type}", flush=True)
    
    print(f"[KEYWORD ENHANCE] Added {added_count} new entities", flush=True)
    print(f"[KEYWORD ENHANCE] Total entities: {len(enhanced)}", flush=True)
    sys.stdout.flush()
    return enhanced


def run_ner_on_text(text, tokenizer, model):
    """
    Split text into sentences, run NER on each, combine with global offsets.
    Then enhance with keyword-based detection.
    Returns (per_sentence_list, combined_entities).
    """
    sentences_with_offsets = split_into_sentences(text)
    if not sentences_with_offsets:
        sentences_with_offsets = [(text.strip(), 0)] if text.strip() else []

    per_sentence = []
    combined = []

    seen_spans = set()  # global (start, end) to avoid duplicate entities in combined

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
                'entity_type': e['entity_type'],
                'start': sent_start + e['start'],
                'end': sent_start + e['end'],
            })
        per_sentence.append({
            'sentence': sent_text,
            'start_offset': sent_start,
            'entities': [
                {'word': e['word'], 'entity_type': e['entity_type'], 'start': e['start'], 'end': e['end']}
                for e in seg_unique
            ],
        })

    # Enhance combined entities with keyword-based detection
    try:
        print(f"[DEBUG] About to enhance. Current count: {len(combined)}")
        combined = enhance_with_keywords(text, combined)
        print(f"[DEBUG] After enhance. New count: {len(combined)}")
    except Exception as e:
        print(f"[ERROR] Keyword enhancement failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Sort by start position
    combined.sort(key=lambda x: x['start'])

    return per_sentence, combined


def load_models():
    """Load the SecBERT NER model and dataset vocabulary"""
    global ner_pipeline, ner_loaded
    global ner_tokenizer, ner_model  # Store separately for custom processing
    global dataset_vocabulary, vocab_loaded
    
    # Load dataset vocabulary
    try:
        print(f"Loading dataset vocabulary from: {VOCABULARY_PATH}")
        if VOCABULARY_PATH.exists():
            with open(VOCABULARY_PATH, 'r', encoding='utf-8') as f:
                dataset_vocabulary = json.load(f)
            
            stats = dataset_vocabulary.get('statistics', {})
            print(f"  Single words: {stats.get('filtered_single_words', 0):,}")
            print(f"  Multi-word entities: {stats.get('filtered_multi_word', 0):,}")
            vocab_loaded = True
            print("[SUCCESS] Dataset vocabulary loaded successfully!")
        else:
            print(f"[WARNING] Vocabulary file not found at {VOCABULARY_PATH}")
            print("[WARNING] Will use hardcoded keywords as fallback")
            vocab_loaded = False
    except Exception as e:
        print(f"[ERROR] Failed to load vocabulary: {e}")
        print("[WARNING] Will use hardcoded keywords as fallback")
        vocab_loaded = False
    
    # Load NER model - using direct model/tokenizer instead of pipeline
    try:
        print(f"Loading NER model from: {NER_MODEL_PATH}")
        ner_tokenizer = AutoTokenizer.from_pretrained(str(NER_MODEL_PATH))
        ner_model = AutoModelForTokenClassification.from_pretrained(str(NER_MODEL_PATH))
        ner_loaded = True
        print("[SUCCESS] NER model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] loading NER model: {e}")
        ner_loaded = False

    return ner_loaded

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    health_info = {
        'status': 'healthy',
        'ner_loaded': ner_loaded,
        'vocabulary_loaded': vocab_loaded
    }
    
    if vocab_loaded and dataset_vocabulary:
        stats = dataset_vocabulary.get('statistics', {})
        health_info['vocabulary_stats'] = {
            'single_words': stats.get('filtered_single_words', 0),
            'multi_word_entities': stats.get('filtered_multi_word', 0),
            'total_entity_types': len(set(
                [v['entity_type'] for v in dataset_vocabulary.get('single_words', {}).values()] +
                [v['entity_type'] for v in dataset_vocabulary.get('multi_word', {}).values()]
            ))
        }
    
    return jsonify(health_info)

@app.route('/api/ner/analyze', methods=['POST'])
def analyze_ner():
    """Analyze text: split into sentences, run NER per sentence, return per-sentence + combined entities."""
    try:
        print("[NER ENDPOINT] Request received")
        if not ner_loaded:
            return jsonify({'error': 'NER model not loaded'}), 500

        data = request.json
        text = data.get('text', '')
        print(f"[NER ENDPOINT] Text: {text}")

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        per_sentence, combined = run_ner_on_text(text, ner_tokenizer, ner_model)
        print(f"[NER ENDPOINT] Combined entities count: {len(combined)}")

        response = {
            'text': text,
            'sentences': per_sentence,
            'entities': combined,
            'entity_count': len(combined),
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("="*70)
    print("SECBERT NER API SERVER")
    print("="*70)

    if load_models():
        print("\nStarting Flask server on http://localhost:5001")
        print("\nAPI Endpoints:")
        print("  NER Model:")
        print("    - POST /api/ner/analyze  - Extract entities")
        print("  General:")
        print("    - GET  /api/health       - Health check")
        print("="*70)

        app.run(debug=False, host='0.0.0.0', port=5001)
    else:
        print("\nFailed to load the NER model!")
        print("Please ensure the fine-tuned SecBERT model is placed at:")
        print(f"  NER: {NER_MODEL_PATH}")
        print("(Run train_secbert_ner.ipynb on Colab to produce it.)")
