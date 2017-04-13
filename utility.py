from nltk import PorterStemmer
from nltk.corpus import stopwords
import string
import math

# Contains some constant variables and term-normalization function.

K = 10
empty_string = ''
COLLECTION_SIZE = "COLLECTION_SIZE"
# Some punctuations that should not be indexed.
punctuations = ["''", '..', '--', '``']

CONTENT_INDEX = "CONTENT_INDEX"
TITLE_INDEX = "TITLE_INDEX"
JURISDICTION = 'JURISDICTION'
COURT = 'COURT'
LANGUAGE = 'english'

stemmer = PorterStemmer()
stopwords = stopwords.words(LANGUAGE)

# Normalizes the given term.
def normalize(term):
    term = term.casefold()
    if term in stopwords:
        return empty_string
    term = stemmer.stem(term)
    if term in string.punctuation or term in punctuations:
        return empty_string
    return term

# Calculates log_tf.
def calculate_log_tf(tf):
    return 1 + math.log(tf, 10)

# Calculates inverse document frequency.
def calculate_idf(collection_size, df):
    return math.log(collection_size/df, 10)
