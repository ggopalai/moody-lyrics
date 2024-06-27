## Models

### Mood classification

- **Model**: BERT (Bidirectional Encoder Representations from Transformers), a pre-trained transformer-based model that achieves state-of-the-art performance across various NLP tasks by capturing bidirectional contextual information from text. Here, the pre-trained model is fine-tuned to take the lyrics as input, and predict one of four moods - happy, relaxed, sad, or angry.
- **Data**: The labelled lyrics dataset - MoodyLyrics is used to train the model. The dataset contains 2,595 songs, all of which are labelled with one of the above moods. Only the song titles and the artists' names were included, since there are copyright restrictions on lyrics. The lyrics were later gathered through the popular song lyric website - Genius, by using its API. We were able to get the lyrics for 2,523 songs.
- **Data Preprocessing**: Lyrics are tokenized using the BERT tokenizer, generating input IDs and attention masks. Special tokens such as [CLS] and [SEP] are added to the begining and the ending of each example, while the [PAD] token is used to maintain a consistent maximum length of model inputs.
- **Training Environment**: The model was trained on Google Colab GPU sing PyTorch. The data was accessed through Google Drive. The pre-trained BERT model was imported from HuggingFace's transformers library. Pandas and Numpy were readily used to help with data transformations and splitting. Off the 2,523 examples - 2018 (80%) were used for training, 253 (10%) for validating, and 252 (10%) for testing the model.
- **Hyperparameters**: The best results were acheived with the following hyperparameter settings, batch size - 32, maximum input length - 256, learning rate - 3 x 10^-5, epochs - 2. Due to memeory constarints, we weren't able to experiment with higher alues for maximum input length.
- **Results**: The model acheieved a 93% accuracy on the test set.
