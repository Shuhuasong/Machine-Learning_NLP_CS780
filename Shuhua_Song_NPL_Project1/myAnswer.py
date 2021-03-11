from myPreprocessing import *
from myModelTrain import*

def main():
    # ************************************************ preprocessing ************************************
    #  -------------------------- Train Data ---------------------------
    fileName1 = 'train.txt'
    outputFile1_train = 'preProcess_train1.txt'
    outputFile2_train = 'preProcess_train2.txt'
    padding_lowerCase(fileName1, outputFile1_train)
    dictFreq_no_unk_train = count_freq_per_word(outputFile1_train)
    #print(sum(dictionary_freq.values()), " ")
    #print(len(dictionary_freq))

    change_singular_word_trainData(outputFile1_train, dictFreq_no_unk_train)
    dictFreq_with_unk_train = count_freq_per_word(outputFile1_train)

    #print(len(dictionary_freq_with_unk))
    #print(dictionary_freq_with_unk['<unk>']);


    # --------------------------- Test Data -----------------------------
    fileName2 = 'test.txt'

    outputFile1_test = "preProcess_test1.txt"
    outputFile2_test = "preProcess_test2.txt"
    padding_lowerCase(fileName2, outputFile1_test)
    dictFreq_no_unk_test = count_freq_per_word(outputFile1_test)
    change_unseen_Word_testData(outputFile1_test, dictFreq_with_unk_train)
    dictFreq_with_unk_test = count_freq_per_word(outputFile1_test)
    bigram_test_with_unk = bigram_model(dictFreq_with_unk_test, outputFile1_test,  False)
    #test_with_unk = bigram_MLE(dictFreq_with_unk_test,outputFile1_train, False)

    #print(dictionary_freq)
    #bigram_model(outputFile1_train, dictionary_freq, True)

    #************************************************ Modeling *******************************************

    # Compute the dictionary of probability by using unigram and bigram

    unigram_dict_train = unigram_model(dictFreq_with_unk_train)
    bigram_dict_train = bigram_model(dictFreq_with_unk_train, outputFile1_train,  False)
    bigram_dict_train_smoothing = bigram_model(dictFreq_with_unk_train, outputFile1_train,  True)
    #bigram_dict_train = bigram_MLE(dictFreq_with_unk_train, outputFile1_train,  False)

    #bigram_MLE(outputFile1_train, dictionary_freq, False)


    # ************************************************ Answer Output *******************************************
    solution_output = open("solution.txt", "w")
    answer = ""
    #Question 1
    answer += ("Question 1: The number of types (unique words, include the padding symbols and the unknown token) in the training corpos are: " + str(len(dictFreq_with_unk_train)) + '\n\n')
    #Question 2
    answer += ("Question 2: The Number of word tokens in the training corpus is : " + str(sum(dictFreq_with_unk_train.values())) + '\n\n')

    #Question 3
    percent_words_test = compute_percentage_token_word(dictFreq_no_unk_train, dictFreq_no_unk_test)
    answer += ("Question 3: " + '\n')
    answer += ('\t\t' + "The percentage of word tokens in test data set but not in training data set is : " + str(round(percent_words_test[0], 4)) + '\n')
    answer += ('\t\t' + "The percentage of word types in test data set but not in training data set is : " + str(round(percent_words_test[1], 4)) + '\n')

    #Question 4
    answer += ("Question 4: " + '\n')
    percent_words_test4 = compute_percentage_token_word(bigram_dict_train[1], bigram_test_with_unk[1])
    answer += ('\t\t' + "The percentage of bigram for tokens in test data set but not in training data set is : " + str(round(percent_words_test4[0], 4)) + '\n')
    answer += ('\t\t' + "The percentage of bigram for types in test data set but not in training data set is : " + str(round(percent_words_test4[1], 4)) + '\n')

    #Question 5 and 6
    answer += ("Question 5 and 6: " + '\n')
    sentence = ["I look forward to hearing your reply ."]
    sentence = mapping_words_unseen(sentence, dictFreq_with_unk_train)
    for sent in sentence:
        answer += get_log_probability_unigram_mode(sent, dictFreq_with_unk_train) + '\n'
        answer += get_log_probability_bigram_mode(sent, bigram_dict_train[1], dictFreq_with_unk_train, False) + '\n' # without Addone
        answer += get_log_probability_bigram_mode(sent, bigram_dict_train[1], dictFreq_with_unk_train, True) + '\n'

    #Question 7
    answer += ("Question 7: " + '\n')
    answer +=  "Compute the perplexity of the entire test corpus under each of the models: " + '\n'
    answer += '\t' + "Perplexity under Unigram model: " + str(get_perplexity_unigram_model(unigram_dict_train, outputFile1_test)) + '\n'
    answer += '\t' + "Perplexity under Bigram model: " + str(get_perplexity_bigram_model(bigram_dict_train, outputFile1_test))+ '\n'
    answer += '\t' + "Perplexity under Bigram model with Add-1 Smoothing: " + str(
           get_perplexity_bigram_model_smoothing(bigram_dict_train[1], dictFreq_with_unk_train, outputFile1_test)) + '\n'


    print(answer)
    solution_output.write(answer)

if __name__ == '__main__':
    main()

