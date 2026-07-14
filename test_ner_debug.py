"""
Debug script to test NER model with the problematic text
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Model path
NER_PATH = r"e:\project\shiley-project\models\secbert_ner_final"

def test_text(text):
    print("\n" + "="*70)
    print("TESTING NER MODEL")
    print("="*70)
    print(f"\nInput text: {text}")
    
    # Load model
    print(f"\nLoading model from: {NER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(NER_PATH)
    model = AutoModelForTokenClassification.from_pretrained(NER_PATH)
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=512)
    offset_mapping = inputs.pop('offset_mapping')[0].tolist()
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)[0].tolist()
    
    # Decode tokens and show predictions
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    print(f"\n{'Token':<25} {'Prediction':<20} {'Offset'}")
    print("-"*70)
    
    detected_entities = []
    for i, (token, pred_id, offset) in enumerate(zip(tokens, predictions, offset_mapping)):
        label = model.config.id2label[pred_id]
        print(f"{token:<25} {label:<20} {offset}")
        
        if label != 'O' and token not in ['[CLS]', '[SEP]', '[PAD]']:
            start, end = offset
            if start < end:
                word = text[start:end]
                entity_type = label.split('-')[-1] if '-' in label else label
                detected_entities.append({
                    'token': token,
                    'word': word,
                    'type': entity_type,
                    'label': label,
                    'offset': offset
                })
    
    print("\n" + "="*70)
    print("DETECTED ENTITIES")
    print("="*70)
    
    if detected_entities:
        for ent in detected_entities:
            print(f"  {ent['word']:<30} -> {ent['type']:<15} (Label: {ent['label']})")
    else:
        print("  No entities detected!")
    
    # Check model labels
    print("\n" + "="*70)
    print("MODEL LABEL MAPPING")
    print("="*70)
    print("\nAvailable labels:")
    for label_id, label_name in model.config.id2label.items():
        print(f"  {label_id}: {label_name}")
    
    return detected_entities

if __name__ == "__main__":
    # Test the problematic text
    test_text1 = "The ransomware Conti targeted healthcare using Microsoft Exchange vulnerabilities"
    entities1 = test_text(test_text1)
    
    print("\n" + "="*70)
    print("\nTesting another example:")
    test_text2 = "APT28 exploited CVE-2023-12345 in a phishing campaign"
    entities2 = test_text(test_text2)
    
    print("\n" + "="*70)
