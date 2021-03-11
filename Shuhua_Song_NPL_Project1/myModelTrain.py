import math
import os

# maximum likelihood model for Unigram  Model
def unigram_model(dict_train):
    dict_train_copy = dict(dict_train)
    dict_train_copy.pop('<s>', None)
    all_tokens = sum(dict_train_copy.values())
    unigram_dict = {}
    for key in dict_train_copy:
        unigram_dict[key] = dict_train_copy[key] / all_tokens
    return unigram_dict


#  maximum likelihood model for bigram model
def bigram_model(dict_train, inputFile, isAddOne):
    dict_train_copy = dict(dict_train)
    bigram_dict = {}
    bigram_posibility_dict = {}
    if os.path.isfile(inputFile):
        readFile = open(inputFile, "r")
        lines = readFile.read()
        for line in lines.splitlines():
            words_line = line.split()
            for i in range(len(words_line) - 1):
                keyWord = words_line[i] + words_line[i+1]
                #keyWord = str(words_line[i]) + str(words_line[i + 1])
                if keyWord in bigram_dict:
                    bigram_dict[keyWord] += 1
                else:
                    bigram_dict[keyWord] = 1
                keyWord = ""
            words_line.clear()
        #print(bigram_dict)
        for line in lines.splitlines():
            words_line = line.split()
            for i in range(len(words_line)-1):
                keyWord = str(words_line[i]) + str(words_line[i+1])
                if(isAddOne==False):
                    bigram_posibility_dict[keyWord] = bigram_dict[keyWord] / dict_train_copy[words_line[i]]
                else:
                    bigram_posibility_dict[keyWord] = (bigram_dict[keyWord] + 1) / (dict_train_copy[words_line[i]] + len(dict_train_copy))
        readFile.close()
    else:
        print("Error!!! no inputFile exists.")
        exit()
    #print(bigram_posibility_dict)
    return [bigram_posibility_dict,bigram_dict]


#Compute log probability for Unigram Model
def get_log_probability_unigram_mode(sentence, dictFreq_with_unk_train):
    log2_probability = 0
    total_log2_prob = 0
    dict_probability_word = {}
    warning_message = ""
    dictFreq_with_unk_train_copy = dict(dictFreq_with_unk_train)
    dictFreq_with_unk_train_copy.pop("<s>")
    sum_tokens = sum(dictFreq_with_unk_train_copy.values())
    new_sentence = sentence.lower() + " </s> "
    answer = "Unigram Model: " + '\n'
    answer += "For Sentence: " + new_sentence + '\n'
    words_sentence = new_sentence.split()
    exist = True
    M = len(words_sentence) # the total number of tokens in the test data
    for word in words_sentence:
        if word in dictFreq_with_unk_train_copy:
            dict_probability_word[word] = dictFreq_with_unk_train_copy[word] / sum_tokens
            log2_probability = math.log2(dict_probability_word[word])

            answer += ("For parameter P(" + str(word) + ") : " + " c(" + str(word) + ") / " + "c(sum_tokens) = "
            + str(dictFreq_with_unk_train_copy[word]) + " / " + str(sum_tokens)
            + "-- the probability is : " + str(dict_probability_word[word])
            + "; the log probability is : " + str(log2_probability) + '\n' )
        else:
            answer += ("The Parameter P(" + str(word) + ") " + "is not exist in the training data" + '\n' )
            exist = False
        if log2_probability == 0:
           warning_message += ("Parameter P(" + str(word) + ") " +  "is 0 "+  '\n')
        else:
            total_log2_prob += log2_probability
        log2_probability = 0
    if exist:
        l = total_log2_prob/M
        perlexity = 2**(-l)
        answer += ( '\n' + "The total log probability is:  " + str(total_log2_prob) + '\n'  )
        answer += ( "The average of total probability is:  " + str(l) + '\n' )
        answer += ( "The perplexity is : " + str(perlexity) + '\n')
    else:
        answer += ( '\n' + "The computation cannot be done, because " + warning_message + '\n' )
    answer += '\n'
    return answer

def get_log_probability_bigram_mode(sentence, bigram_dict_train, dictFreq_with_unk_train, isAddOne):
    new_sentence = "<s> " + sentence.lower() + " </s>"
    answer = ""
    if(isAddOne):
        answer += ( "Bigram Model-Add 1 Smoothing: " + '\n')
    else:
        answer += ( "Bigram Model: " + '\n' )
    answer += ('\t\t' + "For Sentence: " + new_sentence + '\n' )
    dict_probability_word = {}
    total_log2_prob = 0
    words_sentence = new_sentence.split()
    M = len(words_sentence) - 1
    warning_message = ""
    exist = True
    for i in range(len(words_sentence)-1):
        keyword = words_sentence[i] + words_sentence[i+1]
        param_w0 = words_sentence[i]
        param_w1 = words_sentence[i+1]
        givenWord = words_sentence[i]
        log2_probability = 0
        if keyword in bigram_dict_train:
            if givenWord in dictFreq_with_unk_train:
                if isAddOne:
                    dict_probability_word[keyword] = (bigram_dict_train[keyword]+1)/(dictFreq_with_unk_train[givenWord] + len(dictFreq_with_unk_train))
                else:
                    dict_probability_word[keyword] = bigram_dict_train[keyword]/dictFreq_with_unk_train[givenWord]

                log2_probability = math.log2(dict_probability_word[keyword])
                answer += ( "the parameter P(" + str(param_w1) + "|" + str(param_w0) + ") : " + "c(" + str(param_w0) + "," + str(param_w1)
                +  ") / "  +  "c(" + str(param_w0) + ") = "  + str(bigram_dict_train[keyword]) + " / " + str(dictFreq_with_unk_train[givenWord])
                + "; The probability is : " + str(dict_probability_word[keyword])
                + "; The log probability is : " + str(log2_probability) + '\n' )
                total_log2_prob += log2_probability
        else:
            if isAddOne:
                dict_probability_word[keyword] =  1 / (dictFreq_with_unk_train[givenWord] + len(dictFreq_with_unk_train))
                log2_probability = math.log2(dict_probability_word[keyword])
                total_log2_prob += log2_probability
                log_prob_temp = log2_probability
            else:
                exist = False
                warning_message += ( "Parameter P(" + str(param_w1) + " | " + str(param_w0) + ") " + " the log probability is Undefined" + '\n' )
                dict_probability_word[keyword] = 0.0;
                log_prob_temp = "undefined"
            if givenWord in dictFreq_with_unk_train:
                answer += ( "The Parameter P(" + str(param_w1) + " | " + str(param_w0) + ") = " + "c(" + str(param_w0) + "," + str(param_w1)
                + ") = " + str("<unk>") + "/" + "c("  + str(param_w0) + ") = " + str(dictFreq_with_unk_train[givenWord])
                + "; The probability is : " + str(dict_probability_word[keyword])
                + " and the log probability is: " + str(log_prob_temp) + '\n'  )
            else:
                answer += ( "The Parameter P(" + str(param_w1) + " | " + str(param_w0) + ") = " + "c("
                + str(param_w0) + "," + str(param_w1) + ") = "
                + str("<unk>") + "/" + "c(" + str(param_w0) + ") = " + "unk"
                + "; the probability is: " + str(log_prob_temp) + '\n' )
        log2_probability = 0
    if exist:
        l = total_log2_prob / M
        perplexity = 2 ** (-l)
        answer += '\n'
        answer += ( "Total log probability: " + str(total_log2_prob) + '\n' )
        answer += ( "Average of total log probability: " + str(l) + '\n' )
        answer += ( "Perplexity : " + str(perplexity) + '\n' )
    else:
        answer += ( '\n' + "The computation cannot be done : " + warning_message + '\n\n' )
    return answer

#Compute Perplexity for unigram model
def get_perplexity_unigram_model(unigram_dict, inputFile):
    readFile = open(inputFile, "r")
    lines = readFile.read().splitlines()

    total_log2_prob = 0
    M = 0           # the number of tokens
    for line in lines:
        words_line = line.split()
        for i in range(len(words_line)):
            if words_line[i] != '<s>':
                total_log2_prob += math.log2(unigram_dict[words_line[i]])
        M += len(words_line) - 1
    l = total_log2_prob / M
    perplexity = 2**(-l)
    return perplexity

def get_perplexity_bigram_model(bigram_dict, inputFile):
    readFile = open(inputFile, "r")
    lines = readFile.read().splitlines()
    exist = True
    total_log2_prob = 0
    M = 0
    for line in lines:
        words_line = line.split()
        for i in range(len(words_line)-1):
            keyword = words_line[i] + words_line[i+1]
            if keyword in bigram_dict:
                total_log2_prob += math.log2(bigram_dict[keyword])
            else:
                exist = False
        M += len(words_line)
    if exist == False:
        perplexity = ( "We cannot get perplexity because there are some bigrams not found in training data" )
    else:
        l = total_log2_prob/M
        perplexity = 2 ** (-l)
    return perplexity

def get_perplexity_bigram_model_smoothing(bigram_dict_train, dictFreq_with_unk_train, inputFile):
    readFile = open(inputFile, "r")
    lines = readFile.read().splitlines()
    warning_message = ""
    dict_probability_word = {}
    total_log2_prob = 0
    M = 0
    for line in lines:
        words_line = line.split()
        for i in range(len(words_line)-1):
            keyword = words_line[i] + words_line[i+1]
            param_w0 = words_line[i]
            param_w1 = words_line[i+1]
            givenWord = words_line[i]
            log2_probability = 0
            if keyword in bigram_dict_train:
                dict_probability_word[keyword] = (bigram_dict_train[keyword] + 1) / (dictFreq_with_unk_train[givenWord] + len(dictFreq_with_unk_train))
                log2_probability = math.log2(dict_probability_word[keyword])
                total_log2_prob += log2_probability
            else:
                dict_probability_word[keyword] = 1 / (dictFreq_with_unk_train[givenWord] + len(dictFreq_with_unk_train))
                log2_probability = math.log2(dict_probability_word[keyword])
                total_log2_prob += log2_probability

            log2_probability = 0
        M += len(words_line)
        l = total_log2_prob / M
        perplexity = 2 ** (-l)
    return perplexity



def compute_percentage_token_word(dictFreq_no_unk_train, dictFreq_no_unk_test):
    tokens_unseen_sum = 0
    words_unseen_sum = 0
    for word in dictFreq_no_unk_test:
        if word not in dictFreq_no_unk_train:
           tokens_unseen_sum += dictFreq_no_unk_test[word]
           words_unseen_sum += 1;
    tokens_unseen_percent = 100 * (tokens_unseen_sum/sum(dictFreq_no_unk_test.values()))
    words_unseem_percen = 100 * (words_unseen_sum/len(dictFreq_no_unk_test))
    return [tokens_unseen_percent, words_unseem_percen]
