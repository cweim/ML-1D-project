import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
# from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = []
labels = []
current_sentence = []
current_labels = []

# Read the .txt file line by line
with open('RU/train', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if line:  # If the line is not empty
            word, label = line.split()
            current_sentence.append(word)
            current_labels.append(label)
        else:  # Empty line indicates the end of the current sentence
            if current_sentence:  # Ensure it's not an extra empty line
                sentences.append(current_sentence)
                labels.append(current_labels)
                current_sentence = []
                current_labels = []

# Convert labels to sequences of numerical values
sentiment_to_label = {"O": 0, "B-positive": 1, "B-negative": 2, 'B-neutral':3, 'I-positive':4, 'I-neutral':5, 'I-negative':6}  # Map sentiment labels to numerical values
encoded_labels = [[sentiment_to_label[tag] for tag in label] for label in labels]

print(sentiment_to_label)
print(encoded_labels)

# # Now you have 'sentences' as tokenized sentences and 'encoded_labels' as corresponding sentiment labels


# # Assuming you have preprocessed data, word embeddings, and labels

# # Define model parameters
# embedding_dim = 100
# input_length = 103  # Set this to your maximum sequence length
# num_classes = 3  # Number of sentiment classes

# # Padding the sequences to a fixed length
# padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# # Build the CNN model
# model = Sequential()
# model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))
# model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

# # Compile the model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Convert labels to one-hot encoding
# one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

# # Train the model
# model.fit(padded_sequences, one_hot_labels, epochs=10, batch_size=32, validation_split=0.1)

# # Evaluate the model
# evaluation = model.evaluate(padded_dev_sequences, one_hot_dev_labels)
# print("Loss:", evaluation[0])
# print("Accuracy:", evaluation[1])

# # Predict sentiment labels for test data
# predictions = model.predict(padded_test_sequences)

# # Convert predictions to actual labels (e.g., using argmax)
# predicted_labels = np.argmax(predictions, axis=1)

