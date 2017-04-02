#!/usr/bin/python
from nltk import sent_tokenize, word_tokenize
import os
import getopt
import sys
import re
from node import Node
import json
import pickle
import xml.etree.ElementTree as ET
from utility import *

# Builds index for all documents in directory-of-documents and
# writes the dictionary into dictionary-file and the postings into postings-file.

# HashMap<String, HashMap<String, Objects>>
dictionary = dict()

# Builds index for all documents in file_path.
def process_documents(file_path, dictionary_file, postings_file):
    print('building index...')
    all_files = filter(lambda filename: filename != ".DS_Store", os.listdir(file_path))
    collection = []
    for case in all_files:
        collection.append(case[:-4])
    collection.sort()
    doc_length_table = dict()

    for filename in collection:
        (title, content, court, jurisdiction) = parse_xml(file_path, filename)
        (doc_length, term_index_table) = process_content(content)
        title_index_table = process_title(title)
        update_dictionary(filename, term_index_table, title_index_table, court, jurisdiction)
        doc_length_table[filename] = doc_length
    write_to_disk(dictionary_file, postings_file, doc_length_table, collection)
    print('...index is done building')

# parse_xml reads the xml file and gets the useful tags 
def parse_xml(file_path, filename):
    new_file_path = file_path + str(filename) + '.xml'
    xmldoc = ET.parse(new_file_path)
    nodes = xmldoc.findall('str')

    for node in nodes:
        attribute = node.attrib['name']
        if attribute == 'title':
            title = node.text
        elif attribute == 'content':
            content = node.text
        elif attribute == 'court':
            court = node.text

    arr = xmldoc.findall('arr')
    for node in arr:
        attribute = node.attrib['name']
        if attribute == 'jurisdiction':
            jurisdiction = node.findall("str")[0].text

    return title, content, court, jurisdiction

def process_title(title):
    term_index_table = dict()
    index = 0 

    for line in title:
        for sent in sent_tokenize(line):
            for word in word_tokenize(sent):
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
    
    for line in content:
        for sent in sent_tokenize(line):
            for word in word_tokenize(sent):
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
def update_dictionary(doc_ID, term_index_table, title_index_table, court, jurisdiction):
    for term in term_index_table:
        if term not in dictionary:
            dictionary[term] = []
        postings_element = (doc_ID, term_index_table[term])
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
'''
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
'''
process_documents("test/", "dict", "post")
