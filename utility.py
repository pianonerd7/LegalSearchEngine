from nltk import PorterStemmer
import string
import math

# Contains some constant variables and term-normalization function.

K = 10
empty_string = ''
COLLECTION_SIZE = "COLLECTION_SIZE"
stemmer = PorterStemmer()
# Some punctuations that should not be indexed.
punctuations = ["''", '..', '--', '``']

CONTENT_INDEX = "CONTENT_INDEX"
TITLE_INDEX = "TITLE_INDEX"
JURISDICTION = 'JURISDICTION'
COURT = 'COURT'

# Normalizes the given term.
def normalize(term):
    term = stemmer.stem(term).casefold()
    if term in string.punctuation or term in punctuations:
        return empty_string
    return term

# Calculates log_tf.
def calculate_log_tf(tf):
    return 1 + math.log(tf, 10)

# Calculates inverse document frequency.
def calculate_idf(collection_size, df):
    return math.log(collection_size/df, 10)
