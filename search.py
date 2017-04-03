import getopt
import sys
import pickle
from queryParser import *
from node import Node
import math
import heapq
from utility import *

def read_dictionary_to_memory(dictionary_file_path):
    dictionary = None
    doc_length_table = None
    with open(dictionary_file_path, mode="rb") as df:
        data = pickle.load(df)
        dictionary = data[0]
        doc_length_table = data[1]
    return (dictionary, doc_length_table)

def find_posting_in_disk(dictionary, term, postings_file, tag):
    if term in dictionary[tag]:
        offset = dictionary[tag][term].get_pointer() - postings_file.tell()
        postings_file.seek(offset, 1)
        return pickle.loads(postings_file.read(dictionary[tag][term].length))
    else:
        return []

# Runs all queries in file_of_queries on dictionary_file and postings_file_path
# and write results into disk.
def process_queries(dictionary_file, postings_file_path, file_of_queries, output_file_of_results):
    (dictionary, doc_length_table) = read_dictionary_to_memory(dictionary_file)
    queries = parse_query(file_of_queries)
    results = []
    with open(postings_file_path, 'rb') as postings_file:
        for query in queries:
            results.append(calculate_cosine_score(dictionary, doc_length_table, postings_file, query))
    write_to_output(results, output_file_of_results)

# Calculates the cosine score based on the algorithm described in lecture slides.
def calculate_cosine_score(dictionary, doc_length_table, postings_file, query):
    score_dictionary = dict()
    collection_size = dictionary[COLLECTION_SIZE]
    for query_term, query_tf in query.items():
        if query_term not in dictionary:
            continue
        query_df = dictionary[query_term].get_doc_frequency()
        query_weight = calculate_query_weight(query_tf, query_df, collection_size)
        postings = find_posting_in_disk(dictionary, query_term, postings_file)
        update_score_dictionary(postings, score_dictionary, query_weight)
    return get_top_components(score_dictionary, doc_length_table)

# Calculates the ltc weight for the term in query.
def calculate_query_weight(query_tf, query_df, collection_size):
    query_log_tf = calculate_log_tf(query_tf)
    query_idf = calculate_idf(collection_size, query_df)
    return query_log_tf * query_idf

# Calculates the lnc weight for for all documents in postings
# and updates their scores.
def update_score_dictionary(postings, score_dictionary, query_weight):
    for document in postings:
        (doc_id, doc_tf) = document
        doc_weight = calculate_log_tf(doc_tf)
        if doc_id not in score_dictionary:
            score_dictionary[doc_id] = 0
        score_dictionary[doc_id] += query_weight * doc_weight

# Normalizes the score for each document and returns the top K components.
def get_top_components(score_dictionary, doc_length_table):
    score_heap = []
    # score_heap is a min-heap. So we need to negate score to simulate a 'max-heap'.
    for (doc_id, score) in score_dictionary.items():
        normalized_score = score / doc_length_table[doc_id]
        heapq.heappush(score_heap, (-normalized_score, doc_id))
    top_components = heapq.nsmallest(K, score_heap)
    return [x[1] for x in top_components]

def write_to_output(results, output_file_of_results):
    with open(output_file_of_results, mode="w") as of:
        for result in results:
            of.write(format_result(result) + "\n")

def format_result(result):
    return ' '.join(list(map(str, result)))

'''
cosine_score:
initialize a score_dictionary to store the score of each document
// Calculate score:
for each term t in query frequency table:
    fetch postings list for t
    calculate its weight (tf_idf)
    for each document in the postings:
        calculate its weight (log_tf since df = 1)
        update the score of the document in score_dictionary
        (score_dictionary[document] += weight of t * weight of the document
// Normalization:
for each document that appear in score_dictionary:
    normalized score = score[document] / length_of_document <- get from doc_length_table
    update score
// Rank:
Find teh highest 10 scores in the score_dictionary

Question: Should the length of document be the length of the log_tf vector?


Document

log_tf
get postings for every term in the query. list<postings>
get all docs --> list<docs>
            doc1            doc2            doc3            ...
query term  1 + log(tf)     1 + log(tf)     1 + log(tf)
query term  1 + log(tf)     1 + log(tf)     1 + log(tf)
dict<query term, dict<doc, log_tf>>

df
1

tf*idf = log_tf (since it's just tf*idf * 1)

cos_normalization


Query

log_tf
tf table for every word in the query, 1 + log(tf)
dict<term, 1+log(tf)>

idf_df
get the term object from dictionary and get df
            idf
query term  log(n/df)
query term  log(n/df)

dict<query term, idf>

cos_normalization

'''

def usage():
    print ("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

dictionary_file = postings_file = file_of_queries = output_file_of_results = None
try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)
for o, a in opts:
    if o == '-d':
        dictionary_file = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        output_file_of_results = a
    else:
        assert False, "unhandled option"
if dictionary_file == None or postings_file == None or file_of_queries == None or output_file_of_results == None:
    usage()
    sys.exit(2)

process_queries(dictionary_file, postings_file, file_of_queries, output_file_of_results)
