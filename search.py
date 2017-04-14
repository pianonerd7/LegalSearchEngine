import getopt
import sys
import pickle
from queryParser import *
from node import Node
import math
from utility import *
from nltk.corpus import wordnet as wn
from operator import itemgetter

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
    phrases, query, notstemmed_query = parse_query(query_file)
    result_round_1 = get_query_result(query, dictionary, doc_length_table,
        postings_file_path, True, True)
    update_score_by_query_expansion(result_round_1, phrases, query, notstemmed_query, dictionary, doc_length_table, postings_file_path)
    normalized_score = get_normalized_score(result_round_1, doc_length_table)
    update_score_by_fields(normalized_score, dictionary)
    final_result = sorted(normalized_score.items(), key=itemgetter(1), reverse=True)
    write_to_output(final_result, output_file_of_results)

def update_score_by_query_expansion(result, phrases, query, notstemmed_query, dictionary, doc_length_table, postings_file_path):
    new_phrase_query = get_synonyms_for_phrase(phrases)
    for phrase in new_phrase_query[PHRASE]:
        update_query_result_for_synonyms(result, phrase, 0.1, True, dictionary, doc_length_table, postings_file_path)
    update_query_result_for_synonyms(result, new_phrase_query[WORD], 0.1, False, dictionary, doc_length_table, postings_file_path)

    new_word_query = get_synonyms_for_query(query, notstemmed_query)
    for phrase in new_word_query[PHRASE]:
         update_query_result_for_synonyms(result, phrase, 0.05, True, dictionary, doc_length_table, postings_file_path)
    update_query_result_for_synonyms(result, new_word_query[WORD], 0.05, False, dictionary, doc_length_table, postings_file_path)

def update_query_result_for_synonyms(previous_result, synonyms, weight, remove_incomplete, dictionary, doc_length_table, postings_file_path):
    synonyms_result = get_query_result([synonyms], dictionary, doc_length_table, postings_file_path, remove_incomplete, False, previous_result)
    for result in synonyms_result:
        previous_result[result] += synonyms_result[result] * weight

def update_score_by_fields(normalized_score, dictionary):
    for result in normalized_score:
        if result not in dictionary[COURT]:
            continue
        court = dictionary[COURT][result]
        if court == None:
            continue
        if court in courts_weight:
            normalized_score[result] *= courts_weight[court]
        elif len(court) >= 2:
            prefix = court[:2]
            if prefix in courts_weight:
                normalized_score[result] *= courts_weight[prefix]
            subfix = court[-2:]
            if subfix in courts_weight:
                normalized_score[result] *= courts_weight[subfix]
        if result not in dictionary[TAG]:
            continue
        if dictionary[TAG][result]:
            normalized_score[result] *= 1.2

def get_synonyms_for_phrase(phrases):
    default_word_info = {'tf': 1, 'pos': [0]}
    new_phrase_query = {PHRASE: [], WORD: dict()}
    for phrase in phrases:
        phrase = phrase.replace(space, underscore)
        word_synonyms, phrase_synonym = get_synonyms(phrase)
        for synonym in word_synonyms:
            if synonym in new_phrase_query[WORD]:
                word_info = new_phrase_query[WORD][synonym]
                word_info['tf'] += 1
            else:
                word_info = default_word_info
            new_phrase_query[WORD][synonym] = word_info
        for synonym in phrase_synonym:
            if synonym  in new_phrase_query[PHRASE]:
                continue
            new_phrase_query[PHRASE].append(synonym)
    return new_phrase_query

def get_synonyms_for_query(query, notstemmed_query):
    new_word_query = {PHRASE: [], WORD: dict()}
    for index, phrase in enumerate(query):
        for word, word_info in phrase.items():
            get_synonyms_for_word(word, word_info, new_word_query)
        notstemmed_phrase = notstemmed_query[index]
        for word, word_info in notstemmed_phrase.items():
            get_synonyms_for_word(word, word_info, new_word_query)
    return new_word_query

def get_synonyms_for_word(word, word_info, new_word_query):
    word_synonyms, phrase_synonym = get_synonyms(word)
    for synonym in word_synonyms:
        if synonym in new_word_query[WORD]:
            word_info['tf'] += new_word_query[WORD][synonym]['tf']
        new_word_query[WORD][synonym] = word_info
    for new_phrase in phrase_synonym:
        if new_phrase in new_word_query[PHRASE]:
            continue
        new_word_query[PHRASE].append(new_phrase)

def get_synonyms(word):
    word_synonyms = []
    phrase_synonym = []
    syn_set = wn.synsets(word)
    for syn in syn_set:
        for lemma in syn.lemma_names():
            if lemma in word_synonyms or lemma == word:
                continue
            if underscore not in lemma:
                lemma = stem_term(lemma)
                if lemma == empty_string:
                    continue
                word_synonyms.append(lemma)
            else:
                words = lemma.split(underscore)
                phrase_synonym.append(get_new_phrasal_query(words))
    return word_synonyms, phrase_synonym

# Calculates the results of each phrase in the passed query and intersects them to get the result
# of the query.
def get_query_result(query, dictionary, doc_length_table, postings_file_path,
    remove_incomplete, check_position, result_round_1 = None):
    score_dict = None
    position_list_arr = []
    with open(postings_file_path, 'rb') as postings_file:
        for phrase in query:
            (score, position_list) = calculate_cosine_score(dictionary, doc_length_table, postings_file,
                phrase, remove_incomplete, CONTENT_INDEX, result_round_1)
            if score_dict is None:
                score_dict = score
            else:
                score_dict = intersect_score_dicts([score_dict, score])
            if check_position:
                position_list_arr.append(position_list)

    #AND the phrases
    if len(query) > 1:
        if check_position:
            update_score_by_position(score_dict, position_list_arr)

    return score_dict

def update_score_by_position(score_dict, position_list_arr):
    for i in range(0, len(position_list_arr) - 1):
        for j in range(1, len(position_list_arr)):
            answer_10 = get_positional_intersect(position_list_arr[i], position_list_arr[j], 10)
            for doc in answer_10:
                if doc in score_dict:
                    score_dict[doc] *= 1.5
            answer_50 = get_positional_intersect(position_list_arr[i], position_list_arr[j], 50)
            for doc in answer_50:
                if doc in score_dict and doc not in answer_10:
                    score_dict[doc] *= 1.2

# Intersects and returns the score dictionaries.
def intersect_score_dicts(score_dict_arr):
    if len(score_dict_arr) == 0:
        return score_dict_arr
    dict2 = score_dict_arr[0]
    for i in range(1, len(score_dict_arr)):
        dict1 = score_dict_arr[i]
        for key, value in list(dict2.items()):
            if key not in dict1.keys():
                del dict2[key]

    return dict2

# Calculates the cosine score based on the algorithm described in lecture slides.
def calculate_cosine_score(dictionary, doc_length_table, postings_file, phrase,
    remove_incomplete, tag, result_round_1):
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
        update_score_dictionary(postings, score_dictionary, query_weight, result_round_1)

    position_list = dict()
    if remove_incomplete:
        score_dictionary, position_list = remove_unpositional_docs(score_dictionary, phrase, postings_cache)

    return score_dictionary, position_list

# Gets the document IDs that satisfy the positional constraints and removes the rest from the
# passed score_dictionary.
def remove_unpositional_docs(score_dictionary, phrase, postings_cache):
    if len(phrase) < 2:
        term = next(iter(phrase))
        return score_dictionary, postings_cache[term]

    word_pos_arr = []
    for word in phrase.items():
        for position in word[1]["position"]:
            word_pos_arr.append((word[0], position))
    word_pos_arr.sort(key=itemgetter(1))

    doc_id_lists = []

    for i in range(1, len(word_pos_arr)):
        word1 = word_pos_arr[0][0]
        word2 = word_pos_arr[i][0]
        if word1 not in postings_cache or word2 not in postings_cache:
            doc_id_lists = []
        pos_diff = word_pos_arr[i][1] - word_pos_arr[0][1]
        doc_id_lists.append(get_positional_intersect(postings_cache[word1],
            postings_cache[word2], pos_diff))

    intersected_list = intersect_lists(doc_id_lists)
    position_list = list(intersect_score_dicts(doc_id_lists).items())

    new_dict = dict()
    for doc_id in intersected_list:
        if doc_id in score_dictionary:
            new_dict[doc_id] = score_dictionary[doc_id]
    return new_dict, position_list

# Intersects and returns the passed array of lists.
def intersect_lists(doc_id_lists):
    if len(doc_id_lists) == 0:
        return doc_id_lists
    lst = doc_id_lists[0]

    for i in range(1, len(doc_id_lists)):
        lst = list(set(lst) & set(doc_id_lists[i]))

    return lst

# Returns the document IDs that from passed postings lists where the words are
# within pos_diff positions of each other.
def get_positional_intersect(postings1, postings2, pos_diff):
    answer = dict()
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
                        doc_ID = postings1[i][0]
                        if doc_ID not in answer:
                            answer[doc_ID] = []
                        answer[doc_ID].append(positions1[pp1])

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
def update_score_dictionary(postings, score_dictionary, query_weight, result_round_1):
    for document in postings:
        doc_id = document[0]
        if result_round_1 is not None and doc_id not in result_round_1:
            continue
        doc_tf = len(document[1])
        doc_weight = calculate_log_tf(doc_tf)

        if doc_id not in score_dictionary:
            score_dictionary[doc_id] = 0

        score_dictionary[doc_id] += query_weight * doc_weight

# Normalizes the score for each document.
def get_normalized_score(score_dictionary, doc_length_table):
    for (doc_id, score) in score_dictionary.items():
        score_dictionary[doc_id] = score / doc_length_table[doc_id]
    return score_dictionary

def write_to_output(results, output_file_of_results):
    with open(output_file_of_results, mode="w") as of:
        of.write(format_results(results))

def format_results(results):
    return ' '.join(list(map(str, [result[0] for result in results])))

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
