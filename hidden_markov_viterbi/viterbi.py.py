import sys
import numpy as np


def hmm_viterbi(sentence, tags, transition_tab, observation_tab):
    calculation_tab = []
    for word in sentence:
        columns = []
        for tag in tags:
            if len(calculation_tab)==0:
                columns.append([transition_tab[('<s>', tag)] * observation_tab[(tag, word)], -1,'<s>', tag, word])
            if len(calculation_tab)> 0:
                maximum_prob = -1
                point_value = []
                for x, item in enumerate(calculation_tab[-1]):
                    temp_prob = item[0] * transition_tab[(tags[x], tag)] * observation_tab[(tag, word)]
                    if temp_prob > maximum_prob:
                        maximum_prob = temp_prob
                        point_value = [temp_prob, x, tags[x], tag, word]
                columns.append(point_value)
        calculation_tab.append(columns)
    maximum_index = np.argmax([x[0] for x in calculation_tab[-1]])
    print('Probability of Part of speech sequence is :', calculation_tab[-1][maximum_index][0])
    final_result = []
    for row in reversed(calculation_tab):
        final_result.append(row[maximum_index][4] + '_' + row[maximum_index][3])
        maximum_index = row[maximum_index][1]
    return final_result[::-1]


def read_table(file):
    result = {}
    for index, line in enumerate(open(file, "r").readlines()):
        if index == 0:
            tags = line.rstrip().split(',')
        if index>0:
            for loop, word in enumerate(line.rstrip().split(',')):
                if loop == 0:
                    t = word
                else:
                    result[(t, tags[loop])] = float(word)
    return result, tags[1:]


input_sentence = sys.argv[1]
#input_sentence='Janet will back the bill'.split()
print("Input Sentence :",input_sentence)
transition_tab, tags = read_table('Transition_prob.csv')
observation_tab, x = read_table('Observation_lik.csv')
part_of_speech_tags = hmm_viterbi(input_sentence.split(), tags, transition_tab, observation_tab)
print("The tagged sentence:",' '.join(part_of_speech_tags))