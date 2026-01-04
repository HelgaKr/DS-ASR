import os
import re
import unicodedata
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import math

# ============================================================================
# CONFIGURATION
# ============================================================================
"""If you wish to reuse this script for your KenLM training, adjust these variables."""
CSV_PATH = "./metadata.csv"               # The path to your corpus
TEXT_COLUMN = "sentence"                  # Name of the column containing transcriptions
OUTPUT_DIR = "./language_model"           # Where to save your model
LM_NAME = "DS_lm"                         # Your model name
NGRAM_ORDER = 3                           # N-gram order. Choose depending on your corpus size and language features.

# Text preprocessing (should match your ASR model preprocessing!)
CHARS_TO_REMOVE_REGEX = '[\,\?\.\!\;\:]'

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================
def preprocess_text(text):
    text = unicodedata.normalize('NFC', text)
    text = re.sub(CHARS_TO_REMOVE_REGEX, '', text)
    text = text.lower()
    return text

# ============================================================================
# N-GRAM COUNTING
# ============================================================================
def count_ngrams(sentences, max_order):
    ngram_counts = {n: Counter() for n in range(1, max_order + 1)}
    
    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        
        # Add sentence markers
        words = ['<s>'] + words + ['</s>']
        
        for n in range(1, max_order + 1):
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i:i+n])
                ngram_counts[n][ngram] += 1
    
    return ngram_counts

def extract_unigrams(sentences):
    words = set()
    for sentence in sentences:
        words.update(sentence.split())
    return sorted(words)

# ============================================================================
# ARPA FILE GENERATION
# ============================================================================
def write_arpa_file(ngram_counts, output_path, max_order):
    total_unigrams = sum(ngram_counts[1].values())
    vocab_size = len(ngram_counts[1])
    
    lines = []
    
    # DATA SECTION
    lines.append("\\data\\")
    lines.append(f"ngram 1={len(ngram_counts[1]) + 1}")  # +1 for <unk>
    for n in range(2, max_order + 1):
        lines.append(f"ngram {n}={len(ngram_counts[n])}")
    lines.append("")
    
    # UNIGRAMS
    lines.append("\\1-grams:")
    
    # Add <unk> first
    unk_prob = math.log10(1 / (total_unigrams + vocab_size))
    lines.append(f"{unk_prob:.4f}\t<unk>\t0.0000")
    
    for ngram in sorted(ngram_counts[1].keys()):
        count = ngram_counts[1][ngram]
        prob = (count + 1) / (total_unigrams + vocab_size)
        log_prob = math.log10(prob)
        word = ngram[0]
        
        if word == '</s>':
            lines.append(f"{log_prob:.4f}\t{word}")
        else:
            lines.append(f"{log_prob:.4f}\t{word}\t0.0000")
    lines.append("")
    
    # HIGHER ORDER N-GRAMS
    for n in range(2, max_order + 1):
        lines.append(f"\\{n}-grams:")
        
        for ngram in sorted(ngram_counts[n].keys()):
            count = ngram_counts[n][ngram]
            context = ngram[:-1]
            context_count = ngram_counts[n-1].get(context, 1)
            
            prob = (count + 0.1) / (context_count + 0.1 * vocab_size)
            log_prob = math.log10(max(prob, 1e-10))
            
            ngram_str = " ".join(ngram)
            
            if n < max_order:
                lines.append(f"{log_prob:.4f}\t{ngram_str}\t0.0000")
            else:
                lines.append(f"{log_prob:.4f}\t{ngram_str}")
        
        lines.append("")
    
    lines.append("\\end\\")
    
    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines))
        f.write('\n')

def write_unigrams_file(unigrams, output_path):
    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        for word in unigrams:
            f.write(word + '\n')

# ============================================================================
# MAIN
# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading text from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    texts = df[TEXT_COLUMN].tolist()
    print(f"Found {len(texts)} transcriptions")
    
    processed_texts = []
    for t in texts:
        if isinstance(t, str):
            processed = preprocess_text(t)
            if processed.strip():
                processed_texts.append(processed)
    
    # Extract unigrams (just the words)
    unigrams = extract_unigrams(processed_texts)
    
    # Write unigrams file (for pyctcdecode)
    write_unigrams_file(unigrams, unigrams_path)
    
    # Count n-grams
    ngram_counts = count_ngrams(processed_texts, NGRAM_ORDER)
    
    for n in range(1, NGRAM_ORDER + 1):
        print(f"  {n}-grams: {len(ngram_counts[n])}")
    
    # Write ARPA file
    arpa_path = os.path.join(OUTPUT_DIR, f"{LM_NAME}.arpa")
    write_arpa_file(ngram_counts, arpa_path, NGRAM_ORDER)

    # Summary
    print("Training is completed!")
  
if __name__ == "__main__":
    main()
