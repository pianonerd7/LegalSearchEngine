#!/usr/bin/python
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import os
import getopt
import sys
import re
from node import Node
import pickle
import xml.etree.ElementTree as ET
from utility import *
import datetime

# Builds index for all documents in directory-of-documents and
# writes the dictionary into dictionary-file and the postings into postings-file.

# HashMap<String, HashMap<String, Objects>>
dictionary = dict()
# Preinitialize those dictionaries to same indexing time, by avoiding checking whether
# CONTENT_INDEX / COURT / TAG are in the dictionary.
dictionary[CONTENT_INDEX] = dict()
dictionary[COURT] = dict()
dictionary[TAG] = dict()

# Builds index for all documents in file_path.
def process_documents(file_path, dictionary_file, postings_file):
    print('building index...')
    start = datetime.datetime.now()

    collection = [int(filename[:-len(xml)]) for filename in os.listdir(file_path) if filename != ".DS_Store"]
    collection.sort()
    doc_length_table = dict()
    for filename in collection:
        (content, court, tag) = parse_xml(file_path, filename)
        (doc_length, term_index_table) = process_content(content)
        update_dictionary(filename, term_index_table, court, tag)
        doc_length_table[filename] = doc_length
        term_index_table.clear()
    write_to_disk(dictionary_file, postings_file, doc_length_table, len(collection))
    end = datetime.datetime.now()
    print(str(end - start))
    print('...index is done building')

# parse_xml reads the xml file and gets the useful tags
def parse_xml(file_path, filename):
    new_file_path = file_path + str(filename) + xml
    content = court = ''
    tag = False # tag is a boolean attribute to indicate whether the document has tags

    parser = ET.iterparse(new_file_path, events=("start", "end"))
    nodes = iter(parser)
    event, root = next(nodes)

    for event, elem in nodes:
        if event != 'end':
            continue
        if elem.tag == 'str':
            attribute = elem.get('name', "")
            if attribute == 'content':
                content = elem.text
            elif attribute == 'court':
                court = elem.text
        elif elem.tag == 'arr':
            attribute = elem.attrib['name']
            if court == '' and attribute == 'jurisdiction':
                 # If the document does not contain `court`, we store jurisdiction instead.
                court = elem[0].text
            elif attribute == 'tag':
                tag = True
                root.clear()
                break
        root.clear()
    return content, court, tag

# process_content processes the content of given file and computes a term positional index
# table for the content and the length of the content.
def process_content(content):
    term_frequency_table = dict()
    term_index_table = dict()
    index = 0

    for word in word_tokenize(content):
        term = normalize(word)
        if term == empty_string:
            continue
        if term not in term_frequency_table:
            term_frequency_table[term] = 0
            term_index_table[term] = []
        term_frequency_table[term] += 1
        term_index_table[term].append(index)
        index += 1
    doc_length = calculate_doc_length(term_frequency_table.values())
    return (doc_length, term_index_table)

# Calculates the length of the log_tf vector for the document.
def calculate_doc_length(term_frequencies):
    doc_length = 0
    for tf in term_frequencies:
        log_tf = calculate_log_tf(tf)
        doc_length += log_tf * log_tf
    return math.sqrt(doc_length)

# process_title processes the content of given file and computes a term frequency
# table for the tile.
# However, based on our observation, title does not provide much help for the retrieval of
# documents, and will take some title to index it. Therefore, we decide to remove if
# from the production code.
'''
def process_title(title):
    term_index_table = dict()
    index = 0

    for word in word_tokenize(title):
        term = normalize(word)
        if term == empty_string:
            continue
        if term not in term_index_table:
            term_index_table[term] = []
        term_index_table[term].append(index)
        index += 1
    return term_index_table
'''

# update_dictionary takes the doc id, term positional index table, court and tag info
# and updates the global dictionary after processing each document in
# the collection
def update_dictionary(doc_ID, term_index_table, court, tag):
    for term in term_index_table:
        if term not in dictionary[CONTENT_INDEX]:
            dictionary[CONTENT_INDEX][term] = []
        postings_element = (doc_ID, term_index_table[term])
        dictionary[CONTENT_INDEX][term].append(postings_element)

    dictionary[COURT][doc_ID] = court
    dictionary[TAG][doc_ID] = tag

def write_to_disk(dictionary_file, postings_file, doc_length_table, collection_length):
    dict_to_disk = write_post_to_disk(dictionary, postings_file)
    dict_to_disk[COLLECTION_SIZE] = collection_length
    write_dict_to_disk(dict_to_disk, doc_length_table, dictionary_file)

# Writes postings to disk and gets dict_to_disk.
# dict_to_disk has three keys: CONTENT_INDEX, COURT and TAG.
# Values for those keys store information about those corresponding parts.
# dict_to_disk[COURT] is a dictionary, where doc ID are keys, and values are the court of the case.
# dict_to_disk[TAG] is a dictionary, where doc ID are keys, and values are whether the document
# contains tags.
# dict_to_disk[CONTENT_INDEXT] is a dictionary, where terms in content are keys,
# and Node`s that point to the postings are stored as values in the dictionary.
# The tuple in each posting for content represents (doc ID, term positional index table).
def write_post_to_disk(dictionary, postings_file):
    dicts_to_disk = dict()
    with open(postings_file, mode="wb") as pf:
        for tag in dictionary:
            if tag == CONTENT_INDEX:
                dict_to_disk = dict()
                for key in dictionary[tag]:
                    dict_to_disk[key] = Node(key, len(dictionary[tag][key]), pf.tell(), pf.write(pickle.dumps(dictionary[tag][key])))
            else:
                dict_to_disk = dictionary[tag]
            dicts_to_disk[tag] = dict_to_disk
    return dicts_to_disk

# Writes dictionary_file and doc_length_table to disk.
def write_dict_to_disk(dict_to_disk, doc_length_table, dictionary_file):
    with open(dictionary_file, mode="wb") as df:
        data = [dict_to_disk, doc_length_table]
        pickle.dump(data, df)

def usage():
    print ("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")

directory_of_documents = dictionary_file = postings_file = None
try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)
for o, a in opts:
    if o == '-i':
        directory_of_documents = a
    elif o == '-d':
        dictionary_file = a
    elif o == '-p':
        postings_file = a
    else:
        assert False, "unhandled option"
if directory_of_documents == None or dictionary_file == None or postings_file == None:
    usage()
    sys.exit(2)

process_documents(directory_of_documents, dictionary_file, postings_file)
