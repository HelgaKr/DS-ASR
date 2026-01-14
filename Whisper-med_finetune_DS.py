import os
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

torch.cuda.empty_cache()

# ============================================================================
# CONFIGURATION
# ============================================================================
# ============================================================================
# If you wish to reuse this script for your training, adjust these variables.

AUDIO_FOLDER = "./dataset/"                # Folder containing audio files
REPO_NAME = "./whisper-med"                # Name for saving model
MODEL_CHECKPOINT = "openai/whisper-medium"

BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5                       # This learning rate showed the best results for our corpus
WARMUP_STEPS = 200


# ============================================================================
# LOAD AND PREPARE DATASET
# ============================================================================

def stereo_to_mono(batch):
    audio = batch["audio"]
    # Convert stereo to mono by averaging channels
    if len(audio["array"].shape) == 2:
        mono = audio["array"].mean(axis=0)
    else:
        mono = audio["array"]
    batch["audio"]["array"] = mono
    return batch

dataset = load_dataset("audiofolder", data_dir=AUDIO_FOLDER)
dataset = dataset['train']
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset = dataset.map(stereo_to_mono)


# NOTE: I am using the full dataset for training. To use 10% for validation, replace
# the line below with:
#
#   split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
#   train_dataset = split_dataset["train"]
#   eval_dataset = split_dataset["test"]
#
# Then add 'eval_dataset=eval_dataset' to the Trainer initialization.

train_dataset = dataset


# ============================================================================
# CREATE FEATURE EXTRACTOR, TOKENIZER, AND PROCESSOR
# ============================================================================

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)

tokenizer = WhisperTokenizer.from_pretrained(MODEL_CHECKPOINT, task="transcribe")

processor = WhisperProcessor.from_pretrained(MODEL_CHECKPOINT, task="transcribe")


# ============================================================================
# PREPROCESS AUDIO DATA
# ============================================================================

def prepare_dataset(batch):
    audio = batch["audio"]
    # Compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    # Encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

train_dataset = train_dataset.map(prepare_dataset, num_proc=4)

# ============================================================================
# DATA COLLATOR
# ============================================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have different lengths and need different padding
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # Pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Cut bos token if appended in previous tokenization step (it's appended later anyway)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# ============================================================================
# LOAD MODEL
# ============================================================================


model = WhisperForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


# ============================================================================
# TRAINING ARGUMENTS
# ============================================================================

training_args = Seq2SeqTrainingArguments(
    output_dir=REPO_NAME,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    gradient_checkpointing=True,
    fp16=True,
    num_train_epochs=NUM_EPOCHS,
    evaluation_strategy="no",
    logging_strategy="epoch",
    save_total_limit=2,
    save_strategy="epoch",
    predict_with_generate=True,
    report_to=["tensorboard"],
    greater_is_better=False,
)

# ============================================================================
# TRAINER
# ============================================================================

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

# ============================================================================
# TRAIN
# ============================================================================

trainer.train()

# ============================================================================
# SAVE MODEL AND PROCESSOR
# ============================================================================

model.save_pretrained(REPO_NAME)
processor.save_pretrained(REPO_NAME)

torch.cuda.empty_cache()

print("\nTraining complete!")
