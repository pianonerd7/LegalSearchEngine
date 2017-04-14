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
dictionary[CONTENT_INDEX] = dict()
dictionary[COURT] = dict()
dictionary[TAG] = dict()

# Builds index for all documents in file_path.
def process_documents(file_path, dictionary_file, postings_file):
    print('building index...')
    all_files = [filename for filename in os.listdir(file_path) if filename != ".DS_Store"]
    collection = []
    for case in all_files:
        collection.append(case[:-4])
    collection.sort()
    doc_length_table = dict()
    start = datetime.datetime.now()

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
    tag = False

    parser = ET.iterparse(new_file_path)
    for event, elem in parser:
        if elem.tag == 'str':
            attribute = elem.get('name', "")
            if attribute == 'content':
                content = elem.text
            elif attribute == 'court':
                court = elem.text
        elif elem.tag == 'arr':
            attribute = elem.attrib['name']
            if court == '' and attribute == 'jurisdiction':
                court = elem[0].text
            elif attribute == 'tag':
                tag = True
                elem.clear()
                continue
        elem.clear()
    return content, court, tag

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
    with open('1.txt', 'w', encoding='utf8') as dictionary_output:
        dictionary_output.write(str(dict_to_disk))
    dict_to_disk[COLLECTION_SIZE] = collection_length
    write_dict_to_disk(dict_to_disk, doc_length_table, dictionary_file)

# Writes postings to disk and gets dict_to_disk.
# The tuple in each posting represents (doc ID, term freq)
# The keys in dict_to_disk are doc_ids and values are Nodes.
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
