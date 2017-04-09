from nltk import word_tokenize
from utility import *
import re
from nltk.corpus import stopwords
import copy
import string

def parse_query(query_file_path):
    query_arr = query_from_file_to_array(query_file_path)
    return query_arr

def query_from_file_to_array(query_file_path):
    query = []
    f1 = open(query_file_path, 'r', encoding = "utf8")
    line = f1.readline()
    
    # print(line)

    phrases = [f.group(1) for f in re.finditer('"(.+?)"', line)]
    # print(phrases)
    stemmer = PorterStemmer()

    for phrase in phrases:
        words = word_tokenize(phrase)
        pos = 0
        word_dict = dict()
        words = [stemmer.stem(word) for word in words if len(word) > 1 or (word not in punctuations and word not in string.punctuation)]
        for word in words:
            if word not in word_dict:
                word_dict[word] = {"position": [], "tf": 0}

            word_dict[word]["position"].append(pos)
            word_dict[word]["tf"] += 1
            pos += 1

        dict_without_stop_words = copy.deepcopy(word_dict)

        for key in dict_without_stop_words.keys():
            if key in stopwords.words(LANGUAGE):
                del word_dict[key]
            
        query.append(word_dict)

    # print(query)
    positive_docs = []
    negative_docs = []

    for line in f1:
        tokens = line.split()
        if tokens[0] == '+':
            positive_docs.append(int(tokens[1]))
        else:
            negative_docs.append(int(tokens[1]))

    # print(positive_docs)
    # print(negative_docs)
    f1.close()
    return query, positive_docs, negative_docs




# def query_from_file_to_array(query_file_path):
#     queries = []
#     with open(query_file_path, mode="r") as qf:
#         for line in qf:
#             query = line
#             if line[-1:] == "\n":
#                 query = line[:-1]
#             query_freq_table = dict()
#             for term in word_tokenize(query):
#                 term = normalize(term)
#                 if term == empty_string:
#                     continue
#                 if term not in query_freq_table:
#                     query_freq_table[term] = 0
#                 query_freq_table[term] += 1
#             queries.append(query_freq_table)
#     return queries

#parse_query("test/query")
