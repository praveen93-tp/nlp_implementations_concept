import sys

def read_sentences_from_file():
    file = open("NLP6320_POSTaggedTrainingSet-Windows.txt","r")
    dataset = []
    for line in file.readlines():
        dataset.append([token.split('_')[0].lower() for token in line.split()])
    return dataset

class BigramModel:
    def __init__(self,list_of_sentences,smoothing=False,good_Turing=False):
        self.bigram_frequency_count = dict()
        self.unique_bigrams = set()
        self.unigram_frequency_count = dict()
        self.corpus_length=0
        for sentence in list_of_sentences:
            previous_word=None
            for word in sentence:
                self.unigram_frequency_count[word] = self.unigram_frequency_count.get(word,0)+1
                if previous_word != None:
                    self.bigram_frequency_count[(previous_word, word.lower())] = self.bigram_frequency_count.get((previous_word, word),0)+1
                    self.unique_bigrams.add((previous_word, word))
                previous_word = word.lower()
                self.corpus_length = self.corpus_length+1
        self.smoothing=smoothing
        self.vocab = len(self.bigram_frequency_count)
        self.total_bigrams = sum([len(line) - 1 for line in list_of_sentences])

    def calculate_unigram_prob(self):
        unigram_prob = {}
        for key in self.unigram_frequency_count:
            unigram_prob[key] = self.unigram_frequency_count[key]/self.corpus_length
        return unigram_prob

    def get_parameters(self):
        return self.bigram_frequency_count,self.unique_bigrams,self.unigram_frequency_count,self.corpus_length

    def calc_normal_bigram_prob(self):
        bigram_prob = {}
        for bigrams in self.unique_bigrams:
            bigram_prob[bigrams] = (self.bigram_frequency_count.get(bigrams)) / (self.unigram_frequency_count.get(bigrams[0]))
        file = open('bigram_prob.txt', 'w')
        file.write('Bigram' + '\t' + 'Count' + '\t' + 'Probability' + '\n')
        for bigrams in self.unique_bigrams:
            file.write(str(bigrams) + ' : ' + str(self.bigram_frequency_count[bigrams])+ ' : ' + str(bigram_prob[bigrams]) + '\n')
        file.close()
        return bigram_prob

    def calc_bigram_lapalace_smooth_prob(self):
        bigram_smooth_prob = {}
        for bigrams in self.unique_bigrams:
            bigram_smooth_prob[bigrams] = (self.bigram_frequency_count.get(bigrams)+1)/(self.unigram_frequency_count.get(bigrams[0])+len(self.unigram_frequency_count))
        file = open('bigram_lapalace_smooth_prob.txt', 'w')
        file.write('Bigram' + '\t' + 'Count' + '\t' + 'Probability' + '\n')
        for bigrams in self.unique_bigrams:
            file.write(str(bigrams) + ' : ' + str(self.bigram_frequency_count[bigrams]) + ' : ' + str(bigram_smooth_prob[bigrams]) + '\n')
        file.close()
        return bigram_smooth_prob

    def calc_bigram_good_turing(self):
        N_c,c_st,p_st = {},{},{}
        for key in self.bigram_frequency_count:
            if self.bigram_frequency_count[key] in N_c:
                N_c[self.bigram_frequency_count[key]].append(key)
            else:
                N_c[self.bigram_frequency_count[key]]=[key]
        p_st[0] = len(N_c[1])/self.total_bigrams
        for key in N_c:
            if key + 1 in N_c:
                c_st[key] = (key+1)*len(N_c[key + 1])/len(N_c[key])
            else:
                c_st[key] = 0
            p_st[key] = c_st[key] / self.total_bigrams
        return p_st

    def calculate_bigram_sentence_probability(self, sentence,unigram_prob,prob_dict,param='naive'):
        log_sum_res = 1
        previous_word = None
        flag = 0
        if param=='naive':
            for word in sentence.split():
                if flag==0:
                    log_sum_res = log_sum_res*unigram_prob.get(word,0)
                    flag=1
                if log_sum_res==0:
                    break
                else:
                    if previous_word != None:
                        log_sum_res *= prob_dict.get((previous_word, word.lower()),0)
                previous_word = word.lower()

        if param=='smooth':
            for word in sentence.split():
                if flag == 0:
                    log_sum_res = log_sum_res * unigram_prob.get(word.lower(), 0)
                    flag = 1
                if log_sum_res == 0:
                    break
                else:
                    if previous_word != None:
                        log_sum_res *= prob_dict.get((previous_word, word.lower()),  1/(self.unigram_frequency_count[previous_word]+ len(self.unigram_frequency_count)))
                previous_word = word.lower()

        if param=='gturing':
            for word in sentence.split():
                if flag == 0:
                    log_sum_res = log_sum_res * unigram_prob.get(word.lower(), 0)
                    flag = 1
                if log_sum_res == 0:
                    break
                else:
                    if previous_word != None:
                        if (previous_word, word.lower()) in self.bigram_frequency_count:
                            log_sum_res *= prob_dict.get(self.bigram_frequency_count.get((previous_word, word.lower())))
                        else:
                            log_sum_res *= prob_dict.get(0)
                previous_word = word.lower()
        return log_sum_res

    def calculate_unigram_sentence_prob(self,list_of_sentences,unigram_prob):
        log_sum = 1
        for word in list_of_sentences.split():
            log_sum *=  unigram_prob.get(word.lower())
        return log_sum


input_sentence = sys.argv[1]
param = sys.argv[2]
dataset = read_sentences_from_file()
b = BigramModel(dataset,smoothing=True)
unigram_prob = b.calculate_unigram_prob()

#bigram_prob = b.calc_normal_bigram_prob()
#bigram_prob_smooth = b.calc_bigram_lapalace_smooth_prob()
#bigram_gturing = b.calc_bigram_good_turing()

if param == 'naive':
    prob_to_pass = b.calc_normal_bigram_prob()
if param == 'smooth':
    prob_to_pass = b.calc_bigram_lapalace_smooth_prob()
if param == 'gturing':
    prob_to_pass = b.calc_bigram_good_turing()

print("The Resultant probability:",b.calculate_bigram_sentence_probability(input_sentence,unigram_prob,prob_to_pass,param=param))
exit()


