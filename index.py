#!/usr/bin/python
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import os
import getopt
import sys
import re
from node import Node
import json
import pickle
import xml.etree.ElementTree as ET
from utility import *
import copy
import datetime

# Builds index for all documents in directory-of-documents and
# writes the dictionary into dictionary-file and the postings into postings-file.

# HashMap<String, HashMap<String, Objects>>
dictionary = dict()
dictionary[CONTENT_INDEX] = dict()
dictionary[TITLE_INDEX] = dict()
# dictionary[JURISDICTION] = dict()
dictionary[COURT] = dict()

# Builds index for all documents in file_path.
def process_documents(file_path, dictionary_file, postings_file):
    print('building index...')
    all_files = [filename for filename in os.listdir(file_path) if filename != ".DS_Store"]
    doc_length_table = dict()
    start = datetime.datetime.now()

    for case in all_files:
        filename = case[:-4]
        # (title, content, court, jurisdiction) = parse_xml(file_path, case)
        (title, content, court) = parse_xml(file_path, case)
        (doc_length, term_index_table) = process_content(content)
        title_index_table = process_title(title)
        # update_dictionary(filename, term_index_table, title_index_table, court, jurisdiction)
        update_dictionary(filename, term_index_table, title_index_table, court)
        doc_length_table[filename] = doc_length
    write_to_disk(dictionary_file, postings_file, doc_length_table, len(all_files))
    end = datetime.datetime.now()
    print(str(end - start))
    print('...index is done building')

# parse_xml reads the xml file and gets the useful tags
def parse_xml(file_path, filename):
    new_file_path = file_path + filename
    xmldoc = ET.parse(new_file_path)
    nodes = xmldoc.findall('str')
    # title = content = court = jurisdiction = ''
    title = content = court = ''

    for node in nodes:
        attribute = node.attrib['name']
        if attribute == 'title':
            title = node.text
        elif attribute == 'content':
            content = node.text
        elif attribute == 'court':
            court = node.text

    '''for node in xmldoc.iterfind('arr[@name="jurisdiction"]'):
        jurisdiction = node[0].text'''

    # return title, str(content), court, jurisdiction
    return title, content, court

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


# process_content processes the given file and computes a term frequency
# table for that file and the length of the file.
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

# update_dictionary takes the term frequency table as well as the doc id
# and updates the global dictionary after processing each document in
# the collection
# def update_dictionary(doc_ID, term_index_table, title_index_table, court, jurisdiction):
def update_dictionary(doc_ID, term_index_table, title_index_table, court):
    for term in term_index_table:
        if term not in dictionary[CONTENT_INDEX]:
            dictionary[CONTENT_INDEX][term] = []
        postings_element = (doc_ID, term_index_table[term])
        dictionary[CONTENT_INDEX][term].append(postings_element)

    for term in title_index_table:
        if term not in dictionary[TITLE_INDEX]:
            dictionary[TITLE_INDEX][term] = []
        postings_element = (doc_ID, title_index_table[term])
        dictionary[TITLE_INDEX][term].append(postings_element)

    # dictionary[JURISDICTION][doc_ID] = jurisdiction
    dictionary[COURT][doc_ID] = court

def write_to_disk(dictionary_file, postings_file, doc_length_table, collection_length):
    dict_to_disk = write_post_to_disk(dictionary, postings_file)
    dict_to_disk[COLLECTION_SIZE] = collection_length
    write_dict_to_disk(dict_to_disk, doc_length_table, dictionary_file)

# Writes postings to disk and gets dict_to_disk.
# The tuple in each posting represents (doc ID, term freq)
# The keys in dict_to_disk are doc_ids and values are Nodes.
def write_post_to_disk(dictionary, postings_file):
    dicts_to_disk = dict()
    with open(postings_file, mode="wb") as pf:
        for tag in dictionary:
            dict_to_disk = dict()
            for key in dictionary[tag]:
                dict_to_disk[key] = Node(key, len(dictionary[tag][key]), pf.tell(), pf.write(pickle.dumps(dictionary[tag][key])))
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
