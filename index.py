#!/usr/bin/python
from nltk import sent_tokenize, word_tokenize
import os
import getopt
import sys
import re
from node import Node
import json
import pickle
from utility import *

# Builds index for all documents in directory-of-documents and
# writes the dictionary into dictionary-file and the postings into postings-file.

dictionary = dict()

# Builds index for all documents in file_path.
def process_documents(file_path, dictionary_file, postings_file):
    print('building index...')
    all_files = filter(lambda filename: filename != ".DS_Store", os.listdir(file_path))
    collection = list(map(int, all_files))
    collection.sort()
    doc_length_table = dict()

    for filename in collection:
        new_file_path = file_path + str(filename)
        (term_frequency_table, doc_length) = process_document(new_file_path)
        update_dictionary(term_frequency_table, filename)
        doc_length_table[filename] = doc_length
    write_to_disk(dictionary_file, postings_file, doc_length_table, collection)
    print('...index is done building')

# process_document processes the given file and computes a term frequency
# table for that file and the length of the file.
def process_document(file):
    term_frequency_table = dict()

    with open(file, mode="r") as doc:
        for line in doc:
            for sent in sent_tokenize(line):
                for word in word_tokenize(sent):
                    term = normalize(word)
                    if term == empty_string:
                        continue
                    if term not in term_frequency_table:
                        term_frequency_table[term] = 0
                    term_frequency_table[term] += 1
    doc_length = calculate_doc_length(term_frequency_table.values())
    return (term_frequency_table, doc_length)

# Calculates the length of the log_tf vector for the document.
def calculate_doc_length(term_frequencies):
    doc_length = 0
    for tf in term_frequencies:
        log_tf = calculate_log_tf(tf)
        doc_length += log_tf * log_tf
    return math.sqrt(doc_length)

# update_dictionary takes the term frequency table as well as the doc id
# and updates the global dictionary after processing each document in
# the collection
def update_dictionary(term_frequency_table, doc_ID):
    for term in term_frequency_table:
        if term not in dictionary:
            dictionary[term] = []
        postings_element = (doc_ID, term_frequency_table[term])
        dictionary[term].append(postings_element)

def write_to_disk(dictionary_file, postings_file, doc_length_table, collection):
    dict_to_disk = write_post_to_disk(dictionary, postings_file)
    dict_to_disk[COLLECTION_SIZE] = len(collection)
    write_dict_to_disk(dict_to_disk, doc_length_table, dictionary_file)

# Writes postings to disk and gets dict_to_disk.
# The tuple in each posting represents (doc ID, term freq)
# The keys in dict_to_disk are doc_ids and values are Nodes.
def write_post_to_disk(dictionary, postings_file):
    dict_to_disk = dict()
    with open(postings_file, mode="wb") as pf:
        for key in dictionary:
            dict_to_disk[key] = Node(key, len(dictionary[key]), pf.tell(), pf.write(pickle.dumps(dictionary[key])))
    return dict_to_disk

# Writes dictionary_file and doc_length_table to disk.
def write_dict_to_disk(dict_to_disk, doc_length_table, dictionary_file):
    with open(dictionary_file, mode="wb") as df:
        data = [dict_to_disk, doc_length_table]
        pickle.dump(data, df)

def printDict(dictionary):
    for key in dictionary:
        k = key.term + ", " + str(key.frequency)
        print (k, dictionary[key])

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
