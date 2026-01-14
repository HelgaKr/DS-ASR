import os
import json
import re
import torch
import numpy as np
import pandas as pd
import unicodedata
from datasets import Dataset, Audio
from dataclasses import dataclass
from typing import Dict, List, Union
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

#from huggingface_hub import login


# ============================================================================
# CONFIGURATION
# ============================================================================
# If you wish to reuse this script for your training, adjust these variables.
CSV_PATH = "./metadata.csv"                # CSV with columns: 'filename', 'transcription'
AUDIO_FOLDER = "./dataset/"                # Folder containing audio files
REPO_NAME = "./wav2vec2-300m"              # Name for saving model
MODEL_CHECKPOINT = "facebook/wav2vec2-xls-r-300m"

BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 30
LEARNING_RATE = 3e-4
WARMUP_STEPS = 400
SAVE_STEPS = 400
LOGGING_STEPS = 10
WEIGHT_DECAY = 0.01                        # Regularization for stability

# ============================================================================
# LOAD AND PREPARE DATASET
# ============================================================================

df = pd.read_csv(CSV_PATH)

# Create dataset from CSV
data = {
    "audio": [os.path.join(AUDIO_FOLDER, f) for f in df["file_name"]],
    "sentence": df["sentence"].tolist()
}

dataset = Dataset.from_dict(data)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, mono=True))

# Normalize to NFC - important for accurate character counting with diacritics
print("Normalizing text to NFC (composed form)...")

def normalize_unicode(batch):
    batch["sentence"] = unicodedata.normalize('NFC', batch["sentence"])
    return batch

dataset = dataset.map(normalize_unicode)

# ============================================================================
# REMOVAL OF PROBLEMATIC FILES
# ============================================================================
# NOTE: I added this part because some files in our corpus were causing the
# loss to drop to 0. Remove this section if you do not have these issues.

# Filter out files where audio is too short for the text (prevents CTC issues)
def filter_by_ratio(batch):
    audio_length = len(batch["audio"]["array"]) / 16000  # Duration in seconds
    text_length = len(batch["sentence"])
    # Need at least 0.05 seconds per character (after NFC normalization)
    if text_length == 0:
        return False
    return audio_length >= text_length * 0.05

initial_count = len(dataset)
dataset = dataset.filter(filter_by_ratio)
filtered_ratio = initial_count - len(dataset)


# NOTE: I am using the full dataset for training. To use 10% for validation, replace
# the line below with:
#
#   split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
#   train_dataset = split_dataset["train"]
#   eval_dataset = split_dataset["test"]
#
# Then add 'eval_dataset=eval_dataset' to the Trainer initialization.

train_dataset = dataset

print(f"Loaded {len(train_dataset)} training samples")

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================
# Remove special characters
# NOTE: This part can be removed if you did a thorough normalization of the corpus.

chars_to_remove_regex = '[\,\?\.\!\;\:]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch

train_dataset = train_dataset.map(remove_special_characters)

# ============================================================================
# CREATE TOKENIZER
# ============================================================================

def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = train_dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=train_dataset.column_names
)

# Create vocabulary dictionary
vocab_list = list(set(vocab_train["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

# Replace space with pipe for visibility
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

# Add special tokens
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

# Save vocabulary
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

# Create tokenizer
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
    "./",
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|"
)

# ============================================================================
# CREATE FEATURE EXTRACTOR AND PROCESSOR
# ============================================================================

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_CHECKPOINT)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# ============================================================================
# PREPROCESS AUDIO DATA
# ============================================================================

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    batch["labels"] = processor(text=batch["sentence"]).input_ids
    return batch

train_dataset = train_dataset.map(
    prepare_dataset,
    remove_columns=train_dataset.column_names
)

# ============================================================================
# DATA COLLATOR
# ============================================================================

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# ============================================================================
# EARLY STOPPING CALLBACK FOR ZERO LOSS
# ============================================================================
# NOTE: I added this part because some files in our corpus were causing the
# loss to drop to 0. Remove this section if you do not have these issues.

class StopOnZeroLossCallback(TrainerCallback):
    """Stop training when loss reaches 0 to prevent overfitting."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            loss = logs.get("loss")
            if loss is not None and loss <= 0.0:
                print(f"\n{'='*60}")
                print(f"Training stopped: Loss reached {loss:.6f}")
                print(f"{'='*60}\n")
                control.should_training_stop = True
        return control

# ============================================================================
# LOAD MODEL
# ============================================================================

print("Loading model...")

model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_CHECKPOINT,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True,
)

# Reinitialize the CTC head for the new vocabulary
model.lm_head = torch.nn.Linear(
    in_features=model.lm_head.in_features,
    out_features=len(processor.tokenizer),
    bias=False
)
with torch.no_grad():
    model.lm_head.weight.normal_(mean=0.0, std=0.02)

# Freeze feature extractor
model.freeze_feature_extractor()

# ============================================================================
# TRAINING ARGUMENTS
# ============================================================================

training_args = TrainingArguments(
    output_dir=REPO_NAME,
    group_by_length=True,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=NUM_EPOCHS,
    gradient_checkpointing=True,
    fp16=True,
    save_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    logging_first_step=True,
    lr_scheduler_type="cosine",            # Cosine decay for stable training
    optim="adamw_torch",                   # Explicit optimizer
    max_grad_norm=0.5,                     # Gradient clipping for stability
)

# ============================================================================
# TRAINER
# ============================================================================

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=processor.feature_extractor,
    callbacks=[StopOnZeroLossCallback()],
)

# ============================================================================
# TRAIN
# ============================================================================

trainer.train()

# ============================================================================
# SAVE MODEL AND PROCESSOR
# ============================================================================

trainer.save_model(REPO_NAME)
processor.save_pretrained(REPO_NAME)

print(f"\nTraining complete!")
