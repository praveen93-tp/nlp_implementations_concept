import sys

def read_sentences_from_file():
    file = open("NLP6320_POSTaggedTrainingSet-Windows.txt","r")
    dataset = []
    for line in file.readlines():
        dataset.append([token for token in line.split()])
    return dataset

def generate_required_probablities(dataset):
    count = 0
    unigram_count_dict, tag_count, unigram_prob = {}, {}, {}
    unigram_count_dict['<s>'] = len(dataset)
    unigram_count_dict['</s>'] = len(dataset)
    tag_count['<s>'] = len(dataset)
    tag_count['</s>'] = len(dataset)
    for sentences in dataset:
        count += len(sentences)
        for words in sentences:
            unigram_count_dict[words.split("_")[0]] = unigram_count_dict.get(words.split("_")[0], 0) + 1
            tag_count[words.split("_")[1]] = tag_count.get(words.split("_")[1], 0) + 1
    # unigram probabilities
    for word in unigram_count_dict:
        unigram_prob[word] = unigram_count_dict.get(word, 0)/count
    bigram_word_count_dict, bigram_word_prob = {}, {}
    bigram_tag_count_dict, bigram_tag_prob = {}, {}
    bigram_word_tag_count_dict, bigram_word_tag_prob = {}, {}
    for sentences in dataset:
        for index in range(0, len(sentences)):
            if (index == 0):
                bigram_word_count_dict[('<s>', sentences[0].split("_")[0])] = bigram_word_count_dict.get(('<s>', sentences[0].split("_")[0]), 0) + 1
                bigram_tag_count_dict[('<s>', sentences[0].split("_")[1])] = bigram_tag_count_dict.get(('<s>', sentences[0].split("_")[1]), 0) + 1
            if (index + 1 < len(sentences)):
                bigram_word_count_dict[(sentences[index].split("_")[0], sentences[index + 1].split("_")[0])] = bigram_word_count_dict.get((sentences[index].split("_")[0], sentences[index + 1].split("_")[0]), 0) + 1
                bigram_tag_count_dict[(sentences[index].split("_")[1], sentences[index + 1].split("_")[1])] = bigram_tag_count_dict.get((sentences[index].split("_")[1], sentences[index + 1].split("_")[1]), 0) + 1
            else:
                bigram_word_count_dict[(sentences[len(sentences) - 1].split("_")[0], '</s>')] = bigram_word_count_dict.get((sentences[len(sentences) - 1].split("_")[0], '</s>'), 0) + 1
                bigram_tag_count_dict[(sentences[len(sentences) - 1].split("_")[1], '</s>')] = bigram_tag_count_dict.get((sentences[len(sentences) - 1].split("_")[1], '</s>'), 0) + 1
            bigram_word_tag_count_dict[sentences[index]] = bigram_word_tag_count_dict.get(sentences[index], 0) + 1
    # bigram word|word probabilites
    for word in bigram_word_count_dict:
        bigram_word_prob[word[1] + "|" + word[0]] = (bigram_word_count_dict.get(word, 0))/(unigram_count_dict.get(word[0]))
    # bigram tag|tag probabilites
    for word in bigram_tag_count_dict:
        bigram_tag_prob[word[1] + "|" + word[0]] = (bigram_tag_count_dict.get(word, 0))/(tag_count.get(word[0]))
    # bigram word|tag probabilites
    for word in bigram_word_tag_count_dict:
        bigram_word_tag_prob[word.replace('_', '|')] = bigram_word_tag_count_dict.get(word, 0)/(tag_count.get(word.split("_")[1]))
    return bigram_word_tag_prob,bigram_tag_prob

def find_word_to_tags_mapping(bigram_word_tag_prob):
    word_to_tag = {}
    for key in bigram_word_tag_prob:
        if key.split('|')[0] in word_to_tag.keys():
            word_to_tag.get(key.split('|')[0]).add(key.split('|')[1])
        else:
            word_to_tag[key.split('|')[0]] = set()
            word_to_tag[key.split('|')[0]].add(key.split('|')[1])
    return word_to_tag


def pos_tagger_naive_bayes(sentence,bigram_tag_prob,bigram_word_tag_prob,word_to_tag_map):
    result,result_sorted={},{}
    """
       for i in range(0, len(sentence)):
           for eachTag in list(word_to_tag.get(sentence[i])):
               tempTagProb = bigram_word_tag_prob.get(sentence[i] + '|' + eachTag, 0) * bigram_tag_prob.get(eachTag + '|' + previous_max_tag, 0)
               if tempTagProb > arg_max_prob:
                   arg_max_prob = tempTagProb
                   arg_max_tag = eachTag
           tagged_sentence = tagged_sentence + ' ' + sentence[i] + '_' + arg_max_tag.upper()
           previous_max_tag = arg_max_tag
           arg_max_prob = 0
    """
    for index in range(len(sentence)):
        for tag in word_to_tag_map.get(sentence[index]):
            if(index==0):
                result[sentence[index] + '|' + tag] = bigram_word_tag_prob.get(sentence[index] + '|' + tag) * bigram_tag_prob.get(tag + '|' + '<s>')
            elif(index==len(sentence)-1):
                list = {}
                for values in result:
                    list[values+' '+sentence[index]+'|'+tag] = result[values] * bigram_word_tag_prob.get(sentence[index]+'|'+tag,0) * bigram_tag_prob.get(tag+'|' +values.split('|')[len(values.split('|'))-1],0) * bigram_tag_prob.get('</s>'+'|'+ tag,0)
                result.update(list)
            else:
                list = {}
                for values in result:
                    list[values+' '+ sentence[index]+'|'+ tag] = result[values] * bigram_word_tag_prob.get(sentence[index]+'|'+ tag,0)* bigram_tag_prob.get(tag+'|'+ values.split('|')[len(values.split('|'))-1],0)
                result.update(list)
    for res in result:
        flag=1
        for word in sentence:
            if res.find(word)==-1:
                flag=0
        if(flag==1):
            result_sorted[res] = result[res]
    pos_tagged_prob = max(result_sorted.values())
    pos_tagged_sentence= max(result_sorted, key=result_sorted.get)
    return pos_tagged_sentence.replace('|','_'),pos_tagged_prob



input_sentence = sys.argv[1]
#input_sentence = 'John went to work .'
#input_sentence=That_DT may_MD be_VB cold_JJ comfort_NN for_IN Belle_NNP McFall_NNP and_CC 350_CD other_JJ workers_NNS who_WP in_IN October_NNP lost_VBD their_PRP$ jobs_NNS at_IN the_DT Cedartown_NNP plant_NN owned_VBN by_IN Arrow_NNP Shirt_NNP ,_, a_DT unit_NN of_IN Bidermann_NNP International_NNP ._.
#input_sentence = 'That may be cold comfort for Belle McFall and 350 other workers who in October lost their jobs at the Cedartown plant owned by Arrow Shirt , a unit of Bidermann International .'
print("Input sentence is :",input_sentence)
input_sentence = input_sentence.split()
dataset = read_sentences_from_file()
bigram_word_tag_prob,bigram_tag_prob = generate_required_probablities(dataset)
word_to_tag_map = find_word_to_tags_mapping(bigram_word_tag_prob)
#posTagging_Greedy(input_sentence,bigram_tag_prob,bigram_word_tag_prob,word_to_tag_map)
#print("bigram_word_tag_prob",bigram_word_tag_prob)
#print("bigram_tag_prob",bigram_tag_prob)
tagged_sentence,prob = pos_tagger_naive_bayes(input_sentence,bigram_tag_prob,bigram_word_tag_prob,word_to_tag_map)
print("POS Tagged sentence is :",tagged_sentence)
print("Probability:",prob)

exit()




