from nltk import word_tokenize
from utility import *

def parse_query(query_file_path):
    query_arr = query_from_file_to_array(query_file_path)
    return query_arr

def query_from_file_to_array(query_file_path):
    queries = []
    with open(query_file_path, mode="r") as qf:
        for line in qf:
            query = line
            if line[-1:] == "\n":
                query = line[:-1]
            query_freq_table = dict()
            for term in word_tokenize(query):
                term = normalize(term)
                if term == empty_string:
                    continue
                if term not in query_freq_table:
                    query_freq_table[term] = 0
                query_freq_table[term] += 1
            queries.append(query_freq_table)
    return queries

#parse_query("test/query")
