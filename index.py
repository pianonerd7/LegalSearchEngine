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
    print('Please delete existing dictionary, postings and temporary files')
    start = datetime.datetime.now()
    temp_postings_file = "temp_posting"

    collection = [int(filename[:-len(xml)]) for filename in os.listdir(file_path) if filename != ".DS_Store"]
    collection.sort()
    doc_length_table = dict()
    i = 0
    for filename in collection:
        (content, court, tag) = parse_xml(file_path, filename)
        (doc_length, term_index_table) = process_content(content)
        update_dictionary(filename, term_index_table, court, tag)
        doc_length_table[filename] = doc_length
        term_index_table.clear()
        i += 1
        if i > 3500:
            # The collection is too big that it cannot fit into memory.
            # Thus, we will write perioridically write dict to disk,
            # and clear some memory.
            write_temp_dict_to_disk(dictionary_file, temp_postings_file)
            dictionary[CONTENT_INDEX].clear()
            i = 0
    write_temp_dict_to_disk(dictionary_file, temp_postings_file)
    dictionary[CONTENT_INDEX] = merge_dictionary(dictionary_file, postings_file, temp_postings_file)
    dictionary[COLLECTION_SIZE] = len(collection)
    write_dict_to_disk(dictionary, doc_length_table, dictionary_file)
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

# Write the temp content dict to disk.
def write_temp_dict_to_disk(dictionary_file, postings_file):
    temp_dict_to_disk = write_content_post_to_disk(dictionary[CONTENT_INDEX], postings_file)
    with open(dictionary_file, mode="a+b") as df:
        pickle.dump(temp_dict_to_disk, df)

# Writes postings to disk and gets dict_to_disk.
# dict_to_disk[CONTENT_INDEXT] is a dictionary, where terms in content are keys,
# and Node`s that point to the postings are stored as values in the dictionary.
# The tuple in each posting for content represents (doc ID, term positional index table).
def write_content_post_to_disk(content_dictionary, postings_file):
    with open(postings_file, mode="a+b") as pf:
        dict_to_disk = dict()
        for key in content_dictionary:
            dict_to_disk[key] = Node(key, len(content_dictionary[key]), pf.tell(), pf.write(pickle.dumps(content_dictionary[key])))
    return dict_to_disk

# Writes dictionary_file and doc_length_table to disk.
def write_dict_to_disk(dict_to_disk, doc_length_table, dictionary_file):
    with open(dictionary_file, mode="wb") as df:
        data = [dict_to_disk, doc_length_table]
        pickle.dump(data, df)

# Merge dictionaries.
def merge_dictionary(dictionary_file, postings_file, temp_postings_file):
    merged_dictionary = dict()
    with open(dictionary_file, 'rb') as df:
        while True:
            try:
                temp_dictionary = pickle.load(df)
            except EOFError:
                break
            else:
                for key, value in temp_dictionary.items():
                    if key not in merged_dictionary:
                        merged_dictionary[key] = []
                    merged_dictionary[key].append(value)
    with open(temp_postings_file, mode="rb") as tpf, open(postings_file, mode="wb") as pf:
        for key, value in merged_dictionary.items():
            postings = []
            for node in value:
                offset = node.get_pointer() - tpf.tell()
                tpf.seek(offset, 1)
                postings.extend(pickle.loads(tpf.read(node.length)))
            merged_dictionary[key] = Node(key, len(merged_dictionary[key]), pf.tell(), pf.write(pickle.dumps(postings)))
            postings.clear()
    return merged_dictionary

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
