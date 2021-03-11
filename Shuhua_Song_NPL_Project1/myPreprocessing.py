import string
import sys
import os


def padding_lowerCase(inputFile, outputFile):
    if os.path.isfile(inputFile):
        text = ""
        writeFile = open(outputFile, "w")
        with open(inputFile, "r") as fp:
            line = fp.readline()
            while line:
                text = "<s> {} </s> ".format(line.strip()).lower()
                writeFile.writelines(text + '\n')
                line = fp.readline()
        fp.close()
        writeFile.close()
    else:
        print("Error!!! no inputFile exists.")
        exit()


def count_freq_per_word(inputFile):
    dict_freq = {}
    if os.path.isfile(inputFile):
        with open(inputFile, "r") as fp:
            line = fp.readline()
            while line:
                words = line.split()
                for word in words:
                    if word in dict_freq:
                        dict_freq[word] += 1
                    else:
                        dict_freq[word] = 1
                line = fp.readline()
        fp.close()
        #print(dict_freq)
    else:
        print("Error!!! no inputFile exists.")
        exit()
    return dict_freq



def change_singular_word_trainData(inputFile, dict_freq):
    if os.path.isfile(inputFile):
       readFile = open(inputFile, "r")
       lines = readFile.read()
       writeFile = open(inputFile, "w")
       for line in lines.splitlines():
           words_line = line.split()
           #print(words_line)
           for i in range(len(words_line)-1):
               if dict_freq[words_line[i]] == 1:
                  words_line[i] = "<unk>"
           writeFile.write(' '.join(words_line) + '\n')
           words_line.clear()

       writeFile.close()

    else:
         print("Error!!! no inputFile exists.")
         exit()



def change_unseen_Word_testData(inputFile, dictFreq_train):
    if os.path.isfile(inputFile):
        readFile = open(inputFile, "r")
        lines = readFile.read()
        writeFile = open(inputFile, "w")
        for line in lines.splitlines():
            words_line = line.split()
            # print(words_line)
            for i in range(len(words_line)):
                if words_line[i] not in dictFreq_train:
                    words_line[i] = "<unk>"
            writeFile.write(' '.join(words_line) + '\n')
            words_line.clear()
        writeFile.close()
    else:
        print("Error!!! no inputFile exists.")
        exit()


def mapping_words_unseen(sentences_list, freq_train_with_unk):
    list = []
    new_sentence = ""
    for sentence in sentences_list:
        sentence = sentence.lower()
        words = sentence.split()
        for i in range(len(words)):
            if words[i] not in freq_train_with_unk:
                words[i] = "<unk>"
            new_sentence = ' '.join(words)
        list.append(new_sentence)
    return list



def pretty_dict(dict):
    outpt_str = ""
    for key, val in dict.items():
        outpt_str += '"' + str(key) + '" : ' + str(val) + '\n'
    #print(outpt_str)


def pretty_bigram(bigram):
    outpt_str = ""
    for tup in bigram:
        outpt_str += '"' + str[tup[0]] + " " + str(tup[1]) + '" : ' + str(tup[2]) + '\n'
    return outpt_str


def changeNoSeenWord_unk(words, isTrainData):
    for i in range(len(words)):
        if(words[i] not in isTrainData):
            words[i] = "<unk>"



