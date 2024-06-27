# -*- coding: utf-8 -*-
"""Bert-redo

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xVKmJy8iU8NHFsWav2SI2XFRh6QdvWV_

# Transformers for lyric Classification

Imports and Setup
"""

from google.colab import drive
drive.mount('/content/drive')

# !pip install transformers

import torch

# Confirm that the GPU is detected
if torch.cuda.is_available():
  # Get the GPU device name.
  device_name = torch.cuda.get_device_name()
  n_gpu = torch.cuda.device_count()
  print(f"Found device: {device_name}, n_gpu: {n_gpu}")
  device = torch.device("cuda")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pandas as pd
import numpy as np
from tqdm import tqdm
import random

from transformers import BertTokenizer, BertForSequenceClassification

"""Read Data"""

train=pd.read_csv('/content/drive/MyDrive/cse256/project/data/train.csv')
val=pd.read_csv('/content/drive/MyDrive/cse256/project/data/validation.csv')
test=pd.read_csv('/content/drive/MyDrive/cse256/project/data/test.csv')

"""Utility Functions"""

def tokenize_and_format(sentences):
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

  # Tokenize all of the sentences and map the tokens to thier word IDs.
  input_ids = []
  attention_masks = []

  # For every sentence...
  for sentence in sentences:
      # `encode_plus` will:
      #   (1) Tokenize the sentence.
      #   (2) Prepend the `[CLS]` token to the start.
      #   (3) Append the `[SEP]` token to the end.
      #   (4) Map tokens to their IDs.
      #   (5) Pad or truncate the sentence to `max_length`
      #   (6) Create attention masks for [PAD] tokens.
      encoded_dict = tokenizer.encode_plus(
                          sentence,                      # Sentence to encode.
                          add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                          max_length = 256,           # Pad & truncate all sentences.
                          padding = 'max_length',
                          truncation = True,
                          return_attention_mask = True,   # Construct attn. masks.
                          return_tensors = 'pt',     # Return pytorch tensors.
                    )

      # Add the encoded sentence to the list.
      input_ids.append(encoded_dict['input_ids'])

      # And its attention mask (simply differentiates padding from non-padding).
      attention_masks.append(encoded_dict['attention_mask'])
  return input_ids, attention_masks

def get_input_and_labels(df):
  input_ids, attention_masks = tokenize_and_format(df.lyrics.values)
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  labels = torch.tensor(df.mood_encoded.values)
  return input_ids,attention_masks,labels

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

"""Preprocess Data"""

X_train_iids,X_train_ams,y_train=get_input_and_labels(train)

X_val_iids,X_val_ams,y_val=get_input_and_labels(val)
X_test_iids,X_test_ams,y_test=get_input_and_labels(test)

train_set = [(X_train_iids[i], X_train_ams[i], y_train[i]) for i in range(len(y_train))]
val_set = [(X_val_iids[i], X_val_ams[i], y_val[i]) for i in range(len(y_val))]
test_set = [(X_test_iids[i], X_test_ams[i], y_test[i]) for i in range(len(y_test))]

train_text = [train.lyrics.values[i] for i in range(len(y_train))]
val_text = [val.lyrics.values[i] for i in range(len(y_val))]
test_text = [test.lyrics.values[i] for i in range(len(y_test))]

"""Initialize model and train"""

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 4, # The number of output labels.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

model.cuda()

batch_size = 16
optimizer = torch.optim.AdamW(model.parameters(),
                  lr = 3e-5, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8
                )
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, verbose=True, gamma=0.1)
epochs = 5

# function to get validation accuracy
def get_validation_performance(val_set):
    # Put the model in evaluation mode
    model.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0

    num_batches = int(len(val_set)/batch_size) + 1

    total_correct = 0

    for i in range(num_batches):

      end_index = min(batch_size * (i+1), len(val_set))

      batch = val_set[i*batch_size:end_index]

      if len(batch) == 0: continue

      input_id_tensors = torch.stack([data[0] for data in batch])
      input_mask_tensors = torch.stack([data[1] for data in batch])
      label_tensors = torch.stack([data[2] for data in batch])

      # Move tensors to the GPU
      b_input_ids = input_id_tensors.to(device)
      b_input_mask = input_mask_tensors.to(device)
      b_labels = label_tensors.to(device)

      # Tell pytorch not to bother with constructing the compute graph during
      # the forward pass, since this is only needed for backprop (training).
      with torch.no_grad():

        # Forward pass, calculate logit predictions.
        outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
        loss = outputs.loss
        logits = outputs.logits

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the number of correctly labeled examples in batch
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        num_correct = np.sum(pred_flat == labels_flat)
        total_correct += num_correct

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_correct / len(val_set)
    return avg_val_accuracy

# training loop

# For each epoch...
for epoch_i in range(0, epochs):
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode.
    model.train()

    # For each batch of training data...
    num_batches = int(len(train_set)/batch_size) + 1

    for i in tqdm(range(num_batches)):
      end_index = min(batch_size * (i+1), len(train_set))

      batch = train_set[i*batch_size:end_index]

      if len(batch) == 0: continue

      input_id_tensors = torch.stack([data[0] for data in batch])
      input_mask_tensors = torch.stack([data[1] for data in batch])
      label_tensors = torch.stack([data[2] for data in batch])

      # Move tensors to the GPU
      b_input_ids = input_id_tensors.to(device)
      b_input_mask = input_mask_tensors.to(device)
      b_labels = label_tensors.to(device)

      # Clear the previously calculated gradient
      model.zero_grad()

      # Perform a forward pass (evaluate the model on this training batch).
      outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
      loss = outputs.loss
      logits = outputs.logits

      total_train_loss += loss.item()

      # Perform a backward pass to calculate the gradients.
      loss.backward()

      # Update parameters and take a step using the computed gradient.
      optimizer.step()

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set. Implement this function in the cell above.
    print(f"Total loss: {total_train_loss}")
    train_acc = get_validation_performance(train_set)
    print(f"Train accuracy: {train_acc}")
    val_acc = get_validation_performance(val_set)
    print(f"Validation accuracy: {val_acc}")
    # scheduler.step()

print("")
print("Training complete!")

"""Final Evaluation on Test Set"""

test_acc = get_validation_performance(test_set)
print(f"Test accuracy: {test_acc}")

"""Saving the model state for future inference"""

torch.save(model.state_dict(), '/content/drive/MyDrive/cse256/project/models/bert-mood-prediction-1.pt')

"""loading the model again (checking)"""

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 4, # The number of output labels.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
model.load_state_dict(torch.load('/content/drive/MyDrive/cse256/project/models/bert-mood-prediction-1.pt'))
model.cuda()
model.eval()

test_acc = get_validation_performance(test_set)

print(test_acc)

