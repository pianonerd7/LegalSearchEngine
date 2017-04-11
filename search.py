import getopt
import sys
import pickle
from queryParser import *
from node import Node
import math
# import heapq
from utility import *
from nltk.corpus import wordnet as wn

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

# Runs the query in query_file on dictionary_file and postings_file_path
# and write results into disk.
def process_queries(dictionary_file, postings_file_path, query_file, output_file_of_results):
    (dictionary, doc_length_table) = read_dictionary_to_memory(dictionary_file)
    query, positive_docs, negative_docs = parse_query(query_file)
    
    # scores haven't been normalized but leaving that out since scoring will continue in 
    # merge_results_and_do_relevance_feedback_and_stuff()
    result_round_1 = get_query_result(query, dictionary, doc_length_table, 
        postings_file_path, True)
    print("result1", result_round_1)
    
    syns = get_synonyms(query)
    result_round_2 = get_query_result(syns, dictionary, doc_length_table, 
        postings_file_path, False)
    print("result2", result_round_2)

    # results = merge_results_and_do_relevance_feedback_and_stuff(result_round_1, result_round_2, 
    #     positive_docs, negative_docs)
    # write_to_output(results, output_file_of_results)

# Returns the synonyms of a query.
def get_synonyms(query):
    new_query = []
    
    for phrase in query:
        new_phrase = dict()
        for word, word_info in phrase.items():
            new_phrase[word] = word_info
            syn_set = wn.synsets(word)
            synonyms = []
            for syn in syn_set:
                for lemma in syn.lemma_names():
                    if len(synonyms) == 5:
                        break
                    if lemma not in synonyms and lemma != word and '_' not in lemma:
                        synonyms.append(lemma)

            for synonym in synonyms:
                new_phrase[synonym] = word_info

        new_query.append(new_phrase)

    return new_query

# Calculates the results of each phrase in the passed query and intersects them to get the result 
# of the query.
def get_query_result(query, dictionary, doc_length_table, postings_file_path, 
    remove_incomplete):
    score_dict_arr = []
    with open(postings_file_path, 'rb') as postings_file:
        for phrase in query:
            score = calculate_cosine_score(dictionary, doc_length_table, postings_file, 
                phrase, remove_incomplete, CONTENT_INDEX)
            score_dict_arr.append(score)
    
    #AND the phrases
    score_dict = score_dict_arr[0]
    if len(query) > 1:
        score_dict = intersect_score_dicts(score_dict_arr)
    
    return score_dict

# Intersects and returns the score dictionaries.
def intersect_score_dicts(score_dict_arr):
    dict3 = dict()
    for i in range(0, len(score_dict_arr) - 1):
        dict1 = score_dict_arr[i]
        dict2 = score_dict_arr[i + 1]
        
        for key in (dict1.keys() & dict2.keys()):
            for dic in [dict1, dict2]:
                if key in dic:
                    if key not in dict3.keys():
                        dict3[key] = 0
                    dict3[key] += dic[key]
    
    return dict3

# Calculates the cosine score based on the algorithm described in lecture slides.
def calculate_cosine_score(dictionary, doc_length_table, postings_file, phrase, 
    remove_incomplete, tag):
    score_dictionary = dict()
    collection_size = dictionary[COLLECTION_SIZE]
    postings_cache = dict()
    
    for query_term, query_info in phrase.items():
        if query_term not in dictionary[tag]:
            continue
        
        query_df = dictionary[tag][query_term].get_doc_frequency()
        query_weight = calculate_query_weight(query_info["tf"], query_df, collection_size)
        postings = find_posting_in_disk(dictionary, query_term, postings_file, tag)
        postings_cache[query_term] = postings
        update_score_dictionary(postings, score_dictionary, query_weight)
    
    if remove_incomplete:
        score_dictionary = remove_unpositional_docs(score_dictionary, phrase, postings_cache)
    
    return score_dictionary

# Gets the document IDs that satisfy the positional constraints and removes the rest from the 
# passed score_dictionary.
def remove_unpositional_docs(score_dictionary, phrase, postings_cache):
    if len(phrase) < 2:
        return score_dictionary

    word_pos_arr = []
    for word in phrase.items():
        for position in word[1]["position"]:
            word_pos_arr.append((word[0], position))

    doc_id_lists = []
    
    for i in range(1, len(word_pos_arr)):
        word1 = word_pos_arr[0][0]
        word2 = word_pos_arr[i][0]
        pos_diff = word_pos_arr[i][1] - word_pos_arr[0][1]
        doc_id_lists.append(get_positional_intersect(postings_cache[word1], 
            postings_cache[word2], pos_diff))

    intersected_list = intersect_lists(doc_id_lists)
    
    new_dict = dict()
    for doc_id in intersected_list:
        if doc_id in score_dictionary:
            new_dict[doc_id] = score_dictionary[doc_id] 
    
    return new_dict 

# Intersects and returns the passed array of lists.
def intersect_lists(doc_id_lists):
    lst = doc_id_lists[0]

    for i in range(1, len(doc_id_lists)):
        lst = list(set(lst) & set(doc_id_lists[i]))

    return lst

# Returns the document IDs that from passed postings lists where the words are 
# within pos_diff positions of each other. 
def get_positional_intersect(postings1, postings2, pos_diff):
    answer = []
    #postings1
    i = 0
    #postings2
    j = 0

    while i < len(postings1) and j < len(postings2):
        if postings1[i][0] == postings2[j][0]:
            lst = []
            positions1 = postings1[i][1]
            positions2 = postings2[j][1]
            pp1 = 0
            pp2 = 0

            while pp1 < len(positions1):
                while pp2 < len(positions2):
                    if abs(positions1[pp1] - positions2[pp2]) <= pos_diff:
                        lst.append(positions2[pp2])
                    elif positions2[pp2] > positions1[pp1]:
                        break
                    pp2 += 1
                
                for element in lst:
                    if abs(element - positions1[pp1]) <= pos_diff:
                        answer.append(postings1[i][0])

                pp1 += 1
            i += 1
            j += 1
        elif postings1[i][0] < postings2[j][0]:
            i += 1
        else:
            j += 1

    return answer

# Calculates the ltc weight for the term in query.
def calculate_query_weight(query_tf, query_df, collection_size):
    query_log_tf = calculate_log_tf(query_tf)
    query_idf = calculate_idf(collection_size, query_df)
    return query_log_tf * query_idf

# Calculates the lnc weight for for all documents in postings
# and updates their scores.
def update_score_dictionary(postings, score_dictionary, query_weight):
    for document in postings:
        doc_id = document[0]
        doc_tf = len(document[1])
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
    print ("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q query-file -o output-file-of-results")

dictionary_file = postings_file = query_file = output_file_of_results = None
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
        query_file = a
    elif o == '-o':
        output_file_of_results = a
    else:
        assert False, "unhandled option"
if dictionary_file == None or postings_file == None or query_file == None or output_file_of_results == None:
    usage()
    sys.exit(2)

process_queries(dictionary_file, postings_file, query_file, output_file_of_results)
