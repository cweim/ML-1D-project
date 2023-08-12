import tensorflow as tf
import nltk

def tokenize_and_pad(tweet, sequence_length):
  """Tokenizes and pads a tweet.

  Args:
    tweet: The tweet to tokenize and pad.
    sequence_length: The desired length of the padded tweet.

  Returns:
    The tokenized and padded tweet.
  """

  # Tokenize the tweet.
  tokens = nltk.word_tokenize(tweet)

  # Pad the tweet to the desired length.
  padded_tokens = [0] * sequence_length
  for i in range(len(tokens)):
    if i < sequence_length:
      padded_tokens[i] = tokens[i]

  return padded_tokens


def preprocess_training_data(training_data, sequence_length):
  """Preprocesses the training data.

  Args:
    training_data: The training data to preprocess.
    sequence_length: The desired length of the padded tweet.

  Returns:
    A list of tokenized and padded tweets.
  """

  preprocessed_training_data = []
  for tweet in training_data:
    preprocessed_training_data.append(tokenize_and_pad(tweet, sequence_length))

  return preprocessed_training_data


if __name__ == "__main__":
  # Load the training data.
  with open("RU/train", "r") as f:
    training_data = f.readlines()

  # Preprocess the training data.
  preprocessed_training_data = preprocess_training_data(training_data, 30)

  # Save the preprocessed training data.
  with open("preprocessed_training_data.txt", "w") as f:
    for tweet in preprocessed_training_data:
      f.write(" ".join(tweet) + "\n")


# class CNNCRFSentimentModel(tf.keras.Model):
#     def __init__(self, vocab_size, embedding_dim, num_classes):
#         super(CNNCRFSentimentModel, self).__init__()
#         self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
#         self.conv1 = tf.keras.layers.Conv1D(128, 3, activation='relu')
#         self.pool1 = tf.keras.layers.MaxPooling1D(2)
#         self.conv2 = tf.keras.layers.Conv1D(128, 4, activation='relu')
#         self.pool2 = tf.keras.layers.MaxPooling1D(2)
#         self.flatten = tf.keras.layers.Flatten()
#         self.dense = tf.keras.layers.Dense(128, activation='relu')
#         self.crf = tf.keras.layers.CRF(num_classes)

#     def call(self, x):
#         x = self.embedding(x)
#         x = self.conv1(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.pool2(x)
#         x = self.flatten(x)
#         x = self.dense(x)
#         return self.crf(x)

# model = CNNCRFSentimentModel(vocab_size=7881, embedding_dim=128, num_classes=3)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=10)
