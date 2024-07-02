from flask import Flask, request, jsonify, send_from_directory
from lyricsgenius import Genius

import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

mood_map = {
    0: 'Angry',
    1: 'Happy',
    3: 'Sad',
    2: 'Relaxed'
}

# Load your pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 4, # The number of output labels.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
model.load_state_dict(torch.load('backend/models/bert-mood-prediction-1.pt', map_location=torch.device('cpu')))
model.eval()

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
  return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

def get_prediction(iids, ams):
    with torch.no_grad():
        # Forward pass, calculate logit predictions.
        outputs = model(iids,token_type_ids=None,
                        attention_mask=ams)
    logits = outputs.logits.detach().numpy()
    pred_flat = np.argmax(logits, axis=1).flatten()
    return pred_flat[0]

def classify_lyrics(lyrics):
    input_ids, attention_masks = tokenize_and_format([lyrics.replace('\n', ' ')])
    prediction = get_prediction(input_ids, attention_masks)
    mood = ["Angry", "Happy", "Relaxed", "Sad"][prediction]
    return mood

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    song_title = data['title']
    artist_name = data['artist']
    success, lyrics = get_lyrics(song_title, artist_name)
    if success:
        mood = classify_lyrics(lyrics)
        return jsonify({'mood': mood, 'lyrics': lyrics})
    return jsonify({'mood': '-', 'lyrics': lyrics})

def get_lyrics(song_title, artist_name):
    # Implement the lyrics fetching logic here
    # This is a placeholder function
    token='PFl5Jdd01ayEMNqxIkuoAWnA7N9Xw9KqD9BSphLmjQ4IBrJqyaTA9CxKP2k8yJpz'
    genius = Genius(token)
    genius.timeout = 300
    try:
        song = genius.search_song(song_title, artist_name)
        if song == None:
            return False, f"Song not found - {song_title} by {artist_name}"
        lyrics=song.lyrics
        if lyrics.count('-')>200:
            return False, f"Song not found - {song_title} by {artist_name}"
        verses=[]
        for x in lyrics.split('Lyrics')[1][:-7].split('\n'):
            if '[' in list(x) or len(x)==0:
                continue
            verses.append(x.replace("\'","'"))
        return True, '\n'.join(verses)
    except TimeoutError:
        return False, "TIMEOUT"

if __name__ == '__main__':
    app.run(debug=True)
