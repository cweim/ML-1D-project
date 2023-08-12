import sys
import numpy as np
import copy

def get_unique(nested_list):
    flattened_list = [item for sublist in nested_list for item in sublist]
    return sorted(list(set(flattened_list)))

def estimate_e(tokens, labels, unique_labels, unique_tokens):
    e_table = np.zeros((len(unique_labels), len(unique_tokens) + 1))
    for _, (token, label) in enumerate(zip(tokens, labels)):
        for _, (word, pos) in enumerate(zip(token, label)):
            e_table[unique_labels.index(pos)][unique_tokens.index(word)] += 1

    for i in range(len(unique_labels)):
        e_table[i, -1] += 1

    e_table /= e_table.sum(axis=1)[:, np.newaxis]
    return e_table

def viterbi(sentence, unique_labels, unique_tokens, e_table, q_table, unk_token):
    n = len(sentence)
    sentence = [None] + sentence
    m = len(unique_labels)
    pi = np.zeros((n + 2, m))

    for j in range(n):
        if sentence[j + 1] in unique_tokens:
            cur_word = sentence[j + 1]
        else:
            cur_word = unk_token

        for cur_index in range(0, m):
            current_e = e_table[cur_index, unique_tokens.index(cur_word)]
            if j == 0:
                current_q = q_table[0, cur_index]
                pi[j + 1, cur_index] = 1 * current_e * current_q
            else:
                max_prob = 0
                for vIndex in range(0, m):
                    current_q = q_table[vIndex + 1, cur_index]
                    cur_prob = pi[j, vIndex] * current_e * current_q

                    if cur_prob > max_prob:
                        max_prob = cur_prob
                pi[j + 1, cur_index] = max_prob

    max_prob = 0
    for prev_index in range(0, m):
        current_q = q_table[prev_index + 1, -1]
        cur_prob = pi[n, prev_index] * current_q
        if cur_prob > max_prob:
            max_prob = cur_prob
    pi[n + 1, -1] = max_prob

    y_star = [unique_labels.index("O")] * (n + 1)
    max_prob = 0

    for cur_index in range(0, m):
        current_q = q_table[cur_index + 1, -1]
        cur_prob = pi[n, cur_index] * current_q

        if cur_prob > max_prob:
            max_prob = cur_prob
            y_star[n] = cur_index

    for j in range(n - 1, 0, -1):
        max_prob = 0
        for cur_index in range(0, m):
            current_q = q_table[cur_index + 1, y_star[j + 1]]
            cur_prob = pi[j, cur_index] * current_q
            if cur_prob > max_prob:
                max_prob = cur_prob
                y_star[j] = cur_index

    labelled_preds = [unique_labels[y] for y in y_star[1:]]
    return labelled_preds

def viterbi_5thbest(sentence, unique_labels, unique_tokens, e_table, q_table, unk_token):
    rank = 5
    n = len(sentence)
    sentence = [None] + sentence
    m = len(unique_labels)
    pi = np.zeros((n + 2, m, rank))

    for j in range(n):
        if sentence[j + 1] in unique_tokens:
            cur_word = sentence[j + 1]
        else:
            cur_word = unk_token

        for cur_index in range(0, m):
            current_e = e_table[cur_index, unique_tokens.index(cur_word)]
            if j == 0:
                current_q = q_table[0, cur_index]
                pi[j + 1, cur_index, :] = 1 * current_e * current_q
            else:
                max_probs = []
                for prev_index in range(0, m):
                    for r in range(rank):
                        current_q = q_table[prev_index + 1, cur_index]
                        cur_prob = pi[j, prev_index, r] * current_e * current_q

                        max_probs.append(cur_prob)
                max_probs.sort(reverse=True)

                if len(max_probs) > rank:
                    max_probs = max_probs[:rank]
                pi[j + 1, cur_index] = max_probs

    max_probs = []
    for prev_index in range(0, m):
        for r in range(rank):
            current_q = q_table[prev_index + 1, -1]
            cur_prob = pi[-1, prev_index, r] * current_q
            max_probs.append(cur_prob)

    max_probs.sort(reverse=True)
    if len(max_probs) > rank:
        max_probs = max_probs[:rank]
    pi[n + 1, -1] = max_probs

    yxs = np.zeros((n + 1, rank), dtype=int) + unique_labels.index("O")
    max_probs = []

    def take_last(elem):
        return elem[-1]

    for prev_index in range(0, m):
        for r in range(rank):
            current_q = q_table[prev_index + 1, -1]
            cur_prob = pi[-1, prev_index, r] * current_q

            max_probs.append([cur_index, cur_prob])
    max_probs.sort(reverse=True, key=take_last)

    def removeRepeated(lst):
        new = []
        for elem in lst:
            if elem[1] != 0 and elem not in new:
                new.append(elem)
        return new

    max_probs = removeRepeated(max_probs)

    if len(max_probs) > rank:
        max_probs = max_probs[:rank]

    parents = [i[0] for i in max_probs]

    yxs[n, :len(max_probs)] = parents

    for j in range(n - 1, 0, -1):
        max_probs = []
        for yx in yxs[j + 1]:
            for cur_index in range(0, m):
                for r in range(rank):
                    current_q = q_table[cur_index + 1, yx]
                    cur_prob = pi[j, cur_index, r] * current_q

                    max_probs.append([cur_index, cur_prob])

        max_probs.sort(reverse=True, key=take_last)
        max_probs = removeRepeated(max_probs)

        if len(max_probs) > rank:
            max_probs = max_probs[:rank]

        parents = [i[0] for i in max_probs]
        yxs[j, :len(max_probs)] = parents

    labelled_preds = [unique_labels[y] for y in yxs.T[-1][1:]]
    return labelled_preds

def predict_p1(input_path, output_path, unique_labels, unique_tokens, e_table):
    unk_token = "#UNK#"
    predict_label = []
    for sentence in get_test_data(input_path):
        inner_predict = []
        for word in sentence:
            if word not in unique_tokens:
                word = unk_token
                pred_label = e_table[:, -1]
            else:
                pred_label = e_table[:, unique_tokens.index(word)]
            most_likely_label = unique_labels[np.argmax(pred_label)]
            inner_predict.append(most_likely_label)
        predict_label.append(inner_predict)

    with open(output_path, 'w', encoding='utf-8') as outp:
        for _, (token, label) in enumerate(zip(get_test_data(input_path), predict_label)):
            for _, (word, pos) in enumerate(zip(token, label)):
                result = word + " " + pos + "\n"
                outp.write(result)
            outp.write('\n')

def predict_p2(input_path, output_path, unique_labels, unique_tokens, e_table, q_table):
    total_preds = []
    for sentence in get_test_data(input_path):
        preds = viterbi(sentence, unique_labels, unique_tokens, e_table, q_table, unk_token="#UNK#")
        total_preds.append(preds)

    with open(output_path, 'w', encoding='utf-8') as outp:
        for _, (token, label) in enumerate(zip(get_test_data(input_path), total_preds)):
            for _, (word, pos) in enumerate(zip(token, label)):
                result = word + " " + pos + "\n"
                outp.write(result)
            outp.write('\n')

def predict_p3(input_path, output_path, unique_labels, unique_tokens, e_table, q_table):
    total_preds = []
    data = get_test_data(input_path)

    for sentence in data:
        preds = viterbi_5thbest(sentence, unique_labels, unique_tokens, e_table, q_table, unk_token="#UNK#")
        total_preds.append(preds)

    with open(output_path, 'w', encoding='utf-8') as outp:
        for _, (token, label) in enumerate(zip(data, total_preds)):
            for _, (word, pos) in enumerate(zip(token, label)):
                result = word + " " + pos + "\n"
                outp.write(result)
            outp.write('\n')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please make sure you have installed Python 3.4 or above!\nAvailable datasets: ES or RU')
        print("Usage on Windows:  python hmm.py <dataset>")
        print("Usage on Linux/Mac:  python3 hmm.py <dataset>")
        sys.exit()

    directory = f'{sys.argv[1]}/{sys.argv[1]}'

    tokens, labels = [], []
    get_data('RU/train', tokens, labels)
    unique_tokens = get_unique(tokens)
    unique_tokens.append("#UNK#")
    unique_labels = get_unique(labels)
    e_table = estimate_e(tokens, labels, unique_labels, unique_tokens)
    q_table = estimate_q(labels, unique_labels)

    # predict_p1(f'{directory}/dev.in', f'{directory}/dev.p1.out', unique_labels, unique_tokens, e_table)
    # predict_p2(f'{directory}/dev.in', f'{directory}/dev.p2.out', unique_labels, unique_tokens, e_table, q_table)
    predict_p3('RU/dev.in', 'RU/dev.p3.out', unique_labels, unique_tokens, e_table, q_table)

    # print('Evaluate on P1')
    # print(evaluate(directory, 'dev.p1.out'))

    # print('Evaluate on P2')
    # print(evaluate(directory, 'dev.p2.out'))

    # print('Evaluate on P3')
    # print(evaluate("RU/", 'dev.p3.out'))
