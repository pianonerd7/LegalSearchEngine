from nltk import word_tokenize
from utility import *
import re
from nltk.corpus import stopwords
import copy
import string

def parse_query(query_file_path):
    query_arr = query_from_file_to_array(query_file_path)
    return query_arr

# To make sure we can get accurate synonyms later, here we return
# three forms of the query: phrsal query, tokenized but not stemmed query,
# and tokenized and stemmed query (the actual query).
def query_from_file_to_array(query_file_path):
    notstemmed_query = []
    query = []
    f1 = open(query_file_path, 'r', encoding = "utf8")
    line = f1.readline()
    phrases = [f.group(1) for f in re.finditer('"(.+?)"', line)]

    for phrase in phrases:
        phrase = phrase.casefold()
        notstemmed_word_dict, word_dict = parse_phrase(phrase)
        notstemmed_query.append(notstemmed_word_dict)
        query.append(word_dict)

    f1.close()
    return phrases, query, notstemmed_query

# Parses the given phrase.
def parse_phrase(phrase):
    words = word_tokenize(phrase)
    pos = 0
    notstemmed_word_dict = dict()
    word_dict = dict()
    for word in words:
        if check_stopwords(word):
            continue
        add_to_dict(word, pos, notstemmed_word_dict)
        stemmed_word = stem_term(word)
        if stemmed_word == empty_string:
            continue
        add_to_dict(stemmed_word, pos, word_dict)
        pos += 1
    return notstemmed_word_dict, word_dict

# Generates a new phrasal query from those given words.
def get_new_phrasal_query(words):
    pos = 0
    word_dict = dict()
    for word in words:
        stemmed_word = normalize(word)
        if stemmed_word == empty_string:
            continue
        add_to_dict(stemmed_word, pos, word_dict)
        pos += 1
    return word_dict

def add_to_dict(word, pos, word_dict):
    if word not in word_dict:
        word_dict[word] = {POSITION: [], tf: 0}

    word_dict[word][POSITION].append(pos)
    word_dict[word][tf] += 1

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

parse_query("q1.txt")
