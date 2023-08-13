# ML1d Final.ipynb - Entity Recognition using Viterbi Algorithm

This Jupyter Notebook provides an implementation of Entity Recognition using the Viterbi algorithm. The goal of algorithm is to identify entities based on the training labels.

## Getting Started

To use this notebook, follow the steps below:

1. Install Required Libraries: Ensure you have the necessary libraries installed. The following libraries are used:
   - `numpy`: For numerical operations
   - `copy`: For creating deep copies of data

2. Data Preparation: The notebook requires training data and testing data in specific formats. The training data should be formatted as follows:
   ```
   token1 label1
   token2 label2
   ...
   
   token1 label1
   ...
   ```
   The testing data should contain an input file with no labels and a labelled output file.

Alternatively, two sets of training and testing data have been provided, in the RU and ES folders for an example in Russian and in Spanish. Your directory should look like this:
   ```
   ML-1D-Project
   ├── ES
   │   ├── dev.in
   │   ├── dev.out
   │   ├── train
   ├── RU
   │   ├── dev.in
   │   ├── dev.out
   │   ├── train
   └── ML1D Final.ipynb
   └── evalResult.py
   └── README.md
   ```

3. Run Code Cells: Run the code cells in sequential order. The notebook provides functions for data preprocessing, model training, prediction, and evaluation.

## Functions Overview

The following functions are defined in ML1D Final.ipynb

1. **process_data(path, unk_token="#UNK#"):**
   - Reads and processes training data from a given path
   - Returns tokens, labels, unique tokens, and unique labels
   - unk_token is used to account for words not contained in the training data.

2. **dev_open(path):**
   - Opens and processes the testing files containing sentences of the in and out file formats
   - Returns a list of sentences with tokens

3. **get_unique(nested_list):**
   - Extracts number of unique words in the file
   - Returns a list of unique words

4. **estimate_emission_matrix(unique_labels, unique_tokens, tokens, labels, unk_token="#UNK#", k=1):**
   - Estimates emission probabilities for tokens given labels
   - Returns an emission probability table

5. **predict_labels_words(unique_labels, unique_tokens, e_table, test_data, unk_token="#UNK#"):**
   - Predicts labels for test data using emission probabilities only
   - Returns a list of predicted labels

6. **estimate_transition_matrix(unique_labels, labels):**
   - Estimates transition probabilities between labels
   - Returns a transition probability table

7. **viterbi_algorithm(unique_labels, unique_tokens, sentence, e_table, q_table, unk_token):**
   - Predicts labels for test data using both emission and transition probabilities
   - Returns a list of predicted labels

8. **predict_labels_sentences(unique_labels, unique_tokens, q_table, e_table, test_data, unk_token):**
   - Predicts labels for test data using both emission and transition probabilities
   - Returns a list of predicted labels

9. **p1(test_data, predict_label_p1, output_path):**
   - Writes the results of prediction (method p1) to an output file

10. **p2(test_data, predict_label_p2, output_path):**
   - Writes the results of prediction (method p2) to an output file

11. **kthbest_viterbi(e_table, q_table, unique_tokens, unique_labels, unk_token, sentence, num):**
   - Implements the k-th best Viterbi algorithm for multiple predictions
   - Returns a list of predicted labels

12. **p3(input_path, output_path, unique_labels, unique_tokens, e_table, q_table, unk_token, num):**
   - Writes the results of prediction (method p3) to an output file

## Usage

1. If using your own data, specify the paths for training and testing data.
2. Run the code cells in order to train the models and make predictions.
3. Check the generated output files for the predictions.
4. Evaluate the results with evalResult.py (documentation below)

## Note

- This notebook is specifically designed for the task of Named Entity Recognition and assumes a particular input format. Make sure to adapt it to your data and task as needed.
- The training and testing data paths, as well as other parameters, should be configured according to your use case.
- Additional documentation and explanations are provided in the code comments for better understanding.

Please consult the comments in the code for more detailed explanations of each function and its parameters.

# evalResult.py - Evaluation Script for Entity Recognition using Viterbi Algorithm

We have provided an evaluation script named `evalResult.py` that can be used to assess the performance of the trained Viterbi Algorithm Model. The script compares the entities and sentiments predicted by your system with the gold (ground truth) data and calculates various evaluation metrics. The repository should look like this:
   ```
   ML-1D-Project
   └── evalResult.py
   ```

### Usage

You can use the script from the command line by running the following command:

#### In Terminal

   'python3 evalResult.py <<gold>> <<predictions>>'

Where <<gold>> is the ground truth, i.e. the actual sentiments that we are trying to predict, and <<predictions>> is the prediction output, the prediction created by the algorithm. For part 1, the exact evaluation call would be:
   'evalResults.py dev.out dev.p1.out'