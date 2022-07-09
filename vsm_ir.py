import os
import sys

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import xml.etree.ElementTree as ET
import nltk
import math
import json
from collections import defaultdict
from collections import Counter

dictionary = {}  # This dictionary holds all the words, the tf-idf for each file for every word.

tfs = {}
tf_idfs = {}
document_lens = {}

query_dictionary = {}

document_norms = defaultdict(int)  # This dictionary holds the length of all document vectors

corpus = {}  # This dictionary holds the words dictionary ("dictionary") and the document_reference (dictionary of all files and their lengths)


# Extract the desired text from file, tokenize the text, filter stop words and stemming.
def update_dictionary(tokens, doc_id):
    for token in tokens:
        if token not in dictionary:
            dictionary[token] = defaultdict(int)
        dictionary[token][doc_id] += 1


def extract_words(filename):
    try:
        stop_words = set(stopwords.words("english"))
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words("english"))

    tokenizer = RegexpTokenizer(r'\w+')
    ps = PorterStemmer()
    xml_tree = ET.parse(filename)
    root = xml_tree.getroot()
    for child in root.findall("./RECORD"):  # extracts all the text from file.
        txt = ""
        id = 0
        for entry in child:
            if entry.tag == "RECORDNUM":
                id = int(entry.text)
            elif entry.tag == ("TITLE" or "ABSTRACT" or "EXTRACT"):
                txt += str(entry.text) + " "

        txt = txt.lower()
        tokens = tokenizer.tokenize(txt)  # tokenize and filter punctuation.

        tokens = [word for word in tokens if not word in stop_words]  # remove stop words.

        for i in range(len(tokens)):  # stemming
            tokens[i] = ps.stem(tokens[i])

        update_dictionary(tokens, id)
        document_lens[id] = len(tokens)
        # get maximum token occurrence
        c = Counter(tokens)
        max_occurrence = c.most_common(1)[0][1]
        tf = {}

        # calculate tfs for each token in the document
        for token in set(tokens):
            tf[token] = c[token] / max_occurrence

        tfs[id] = tf


def calc_tf_idf():
    num_of_documents = len(document_lens)
    for token, tokenDict in dictionary.items():
        idf = math.log2(num_of_documents / len(tokenDict))
        for doc in tokenDict:
            tf = tfs[doc][token]
            w = tf * idf
            tf_idfs[(doc, token)] = w
            document_norms[doc] += w ** 2


def create_index():
    input_dir = sys.argv[2]
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".xml"):
            file = input_dir + "/" + file_name
            extract_words(file)

    calc_tf_idf()

    # Add dictionary and document_reference to corpus
    corpus["dictionary"] = dictionary
    corpus["document_norms"] = document_norms
    corpus["document_lens"] = document_lens

    inverted_index_file = open("vsm_inverted_index.json", "w")
    json.dump(corpus, inverted_index_file, indent=8)
    inverted_index_file.close()


create_index()
