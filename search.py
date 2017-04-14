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
# The score for a document is determined by four aspects:
# tf-idf of the content,
# query expansion,
# distance between every two query phrases in the content of a document,
# and fields info.
def process_queries(dictionary_file, postings_file_path, query_file, output_file_of_results):
    (dictionary, doc_length_table) = read_dictionary_to_memory(dictionary_file)
    phrases, query, notstemmed_query = parse_query(query_file)

    (score_dict, position_list_arr) = get_query_result(query, dictionary, doc_length_table,
        postings_file_path, True, True)
    update_score_by_query_expansion(score_dict, phrases, notstemmed_query, dictionary, doc_length_table, postings_file_path)
    normalized_score = get_normalized_score(score_dict, doc_length_table)
    update_score_by_position(normalized_score, position_list_arr)
    update_score_by_fields(normalized_score, dictionary)
    final_result = sorted(normalized_score.items(), key=itemgetter(1), reverse=True)

    write_to_output(final_result, output_file_of_results)

# Calculates the results of each phrase in the passed query and intersects them to get the result
# of the query.
def get_query_result(query, dictionary, doc_length_table, postings_file_path,
    remove_incomplete, check_position, previous_result = None):
    score_dict = None
    position_list_arr = []
    with open(postings_file_path, 'rb') as postings_file:
        for phrase in query:
            (score, position_list) = calculate_cosine_score(dictionary, doc_length_table, postings_file,
                phrase, remove_incomplete, CONTENT_INDEX, previous_result)
            if score_dict is None:
                score_dict = score
            else:
                score_dict = intersect_dicts(score_dict, score)
            if check_position:
                position_list_arr.append(position_list)

    if check_position:
        return score_dict, position_list_arr

    return score_dict

# To get synonyms of the original query, here we are finding the synonyms of
# the original query phrases and synonyms for those terms appeared in the
# not-stemmed query, by using wordnet.
# It is necessary to do so to ensure to get correct synonyms, because the synonyms
# of a word will depends on its form.
# Synonyms returned by wordnet has two forms: words and phrases.
# For a phrase returned by wordnet, we need to do phrasal queries.
# While for a term, we just consider them as "OR" relations, and perform
# the standard tf-idf search.
# Therefore, there are four difference cases in total.
def update_score_by_query_expansion(result, phrases, notstemmed_query, dictionary, doc_length_table, postings_file_path):
    new_phrase_query = get_synonyms_for_phrase(phrases)
    for phrase in new_phrase_query[PHRASE]:
        update_query_result_for_synonyms(result, phrase, synonyms_phrase_weight, True, dictionary, doc_length_table, postings_file_path)
    update_query_result_for_synonyms(result, new_phrase_query[WORD], synonyms_phrase_weight, False, dictionary, doc_length_table, postings_file_path)

    new_word_query = get_synonyms_for_query(notstemmed_query)
    for phrase in new_word_query[PHRASE]:
         update_query_result_for_synonyms(result, phrase, synonyms_word_weight, True, dictionary, doc_length_table, postings_file_path)
    update_query_result_for_synonyms(result, new_word_query[WORD], synonyms_word_weight, False, dictionary, doc_length_table, postings_file_path)

# Adds score of querying those synonyms multplied by weight to the previous_result.
def update_query_result_for_synonyms(previous_result, synonyms, weight, remove_incomplete, dictionary, doc_length_table, postings_file_path):
    synonyms_result = get_query_result([synonyms], dictionary, doc_length_table, postings_file_path, remove_incomplete, False, previous_result)
    for result in synonyms_result:
        previous_result[result] += synonyms_result[result] * weight

# Normalizes the score for each document.
def get_normalized_score(score_dictionary, doc_length_table):
    for (doc_id, score) in score_dictionary.items():
        score_dictionary[doc_id] = score / doc_length_table[doc_id]
    return score_dictionary

# Updates score by the distance between query terms.
# If query terms are closed to each other, it is more likely that a document
# is relevant.
# Here we set two thresholds: distance within 10 words (same sentence),
# and distance within 50 words (same paragraph).
def update_score_by_position(score_dict, position_list_arr):
    for i in range(0, len(position_list_arr) - 1):
        for j in range(i + 1, len(position_list_arr)):
            answer_100 = get_positional_intersect(position_list_arr[i], position_list_arr[j], 100)
            answer_20 = get_positional_intersect(position_list_arr[i], position_list_arr[j], 20)
            for doc in answer_100:
                if doc in score_dict and doc in answer_20:
                    score_dict[doc] *= 1.5
                elif doc in score_dict:
                    score_dict[doc] *= 1.2

# Updates score by those fields in the document (court and tag).
# We rank the weight for courts: SGCA > SGHC > SG** (any case in SG) > **CA
# (any case happened in the court of appeal of a country) > other
# For example, we suppose that a document is more likely to be relevant if
# the court is SGCA.
# Alo, as stated in the instruction for the assignment, documents with tagged
# are considered as landmark cases and are morelikly to be relevant.
def update_score_by_fields(normalized_score, dictionary):
    for result in normalized_score:
        if result not in dictionary[COURT]:
            continue
        court = dictionary[COURT][result]
        if court == None:
            continue
        if court in courts_score:
            normalized_score[result] += courts_score[court]
        elif len(court) >= 2:
            prefix = court[:2] # for exanple, `SG`
            if prefix in courts_score:
                normalized_score[result] += courts_score[prefix]
            subfix = court[-2:]  # for example, `CA`
            if subfix in courts_score:
                normalized_score[result] += courts_score[subfix]
        if result not in dictionary[TAG]:
            continue
        if dictionary[TAG][result]:
            normalized_score[result] += tag_score

# Retunrs synonyms of phrases in a query.
def get_synonyms_for_phrase(phrases):
    default_word_info = {tf: 1, POSITION: [0]}
    new_phrase_query = {PHRASE: [], WORD: dict()}
    for phrase in phrases:
        if ' ' not in phrase:
            continue
        phrase = phrase.replace(space, underscore)
        word_synonyms, phrase_synonym = get_synonyms(phrase)
        for synonym in word_synonyms:
            if synonym in new_phrase_query[WORD]:
                word_info = new_phrase_query[WORD][synonym]
                word_info[tf] += 1
            else:
                word_info = default_word_info
            new_phrase_query[WORD][synonym] = word_info
        for synonym in phrase_synonym:
            if synonym  in new_phrase_query[PHRASE]:
                continue
            new_phrase_query[PHRASE].append(synonym)
    return new_phrase_query

# Returns synonyms of terms in a query.
def get_synonyms_for_query(notstemmed_query):
    new_word_query = {PHRASE: [], WORD: dict()}
    for phrase in notstemmed_query:
        for word, word_info in phrase.items():
            get_synonyms_for_word(word, word_info, new_word_query)
    return new_word_query

# Returns synonyms of the word.
def get_synonyms_for_word(word, word_info, new_word_query):
    word_synonyms, phrase_synonym = get_synonyms(word)
    for synonym in word_synonyms:
        if synonym in new_word_query[WORD]:
            word_info[tf] += new_word_query[WORD][synonym][tf]
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

# Intersects and returns the score dictionaries.
def intersect_dicts(dict1, dict2):
    for key, value in list(dict1.items()):
        if key not in dict2.keys():
            del dict1[key]

    return dict1

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
        for position in word[1][POSITION]:
            word_pos_arr.append((word[0], position))
    word_pos_arr.sort(key=itemgetter(1))

    doc_id_lists = []

    for i in range(1, len(word_pos_arr)):
        word1 = word_pos_arr[0][0]
        word2 = word_pos_arr[i][0]
        if word1 not in postings_cache or word2 not in postings_cache:
            return dict(), []
        pos_diff = word_pos_arr[i][1] - word_pos_arr[0][1]
        doc_id_lists.append(get_positional_intersect(postings_cache[word1],
            postings_cache[word2], pos_diff))

    intersected_list = intersect_lists(doc_id_lists)
    position_list = list(intersected_list.items())
    position_list.sort(key=itemgetter(0))

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

def write_to_output(results, output_file_of_results):
    with open(output_file_of_results, mode="w") as of:
        of.write(format_results(results))

def format_results(results):
    return ' '.join(list(map(str, [result[0] for result in results])))

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
