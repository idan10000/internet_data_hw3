import os
import sys

import numpy as np
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
bm25_idfs = {}
document_lens = {}

document_norms = defaultdict(float)  # This dictionary holds the length of all document vectors


k = 1.5
b = 0.75


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
            elif entry.tag == "TITLE" or entry.tag == "ABSTRACT" or entry.tag =="EXTRACT":
                txt += str(entry.text) + " "

        txt = txt.lower()
        tokens = tokenizer.tokenize(txt)  # tokenize and filter punctuation.

        tokens = [word for word in tokens if word not in stop_words]  # remove stop words.

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
        bm25_idfs[token] = np.log((len(document_lens) - len(tokenDict) + 0.5) / (len(tokenDict) + 0.5) + 1)
        for doc in tokenDict:
            tf = tfs[doc][token]
            w = tf * idf
            if doc not in tf_idfs:
                tf_idfs[doc] = {}
            tf_idfs[doc][token] = w
            document_norms[doc] += w ** 2
    for doc in document_norms:
        document_norms[doc] = math.sqrt(document_norms[doc])

def create_index():
    input_dir = sys.argv[2]
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".xml"):
            file = input_dir + "/" + file_name
            extract_words(file)

    document_lens["avg"] = sum(document_lens) / len(document_lens)

    calc_tf_idf()
    corpus = {}
    # Add dictionary and document_reference to corpus
    corpus["dictionary"] = dictionary
    corpus["tf_idfs"] = tf_idfs
    corpus["document_norms"] = document_norms
    corpus["document_lens"] = document_lens
    corpus["bm25_idfs"] = bm25_idfs

    inverted_index_file = open("vsm_inverted_index.json", "w")
    json.dump(corpus, inverted_index_file, indent=8)
    inverted_index_file.close()


def preprocess_query():
    query = sys.argv[4].lower()
    stop_words = set(stopwords.words("english"))
    tokenizer = RegexpTokenizer(r'\w+')
    ps = PorterStemmer()
    tokens = tokenizer.tokenize(query)  # tokenize and filter punctuation.
    tokens = [word for word in tokens if word not in stop_words]  # remove stop words.
    for i in range(len(tokens)):  # stemming
        tokens[i] = ps.stem(tokens[i])
    return tokens


def find_relevant_documents(query, dictionary):
    docs = set()
    for token in query:
        if token in dictionary:
            for document in dictionary[token]:
                docs.add(document)
    return docs


def query_bm_25(query, relevant_documents, dictionary, document_lens, bm25_idfs):
    documents = []
    for document in relevant_documents:
        score = 0
        for token in query:
            if token in dictionary:
                idf = bm25_idfs[token]
                if document in dictionary[token]:
                    frequency = dictionary[token][document]
                else:
                    frequency = 0
                score += idf * (frequency * (k + 1)) / (
                        frequency + k * (1 - b + (b * document_lens[document] / document_lens["avg"])))
        documents.append((document, score))
    documents.sort(key=lambda t: t[1], reverse=True)
    return documents


def calc_query_tf_idf(query, dictionary, amount_of_docs):
    query_dictionary = {}
    counter = Counter(query)
    max_occurrence = counter.most_common(1)[0][1]
    for token in query:

        tf = counter[token] / max_occurrence
        if token in dictionary:
            idf = math.log2(amount_of_docs / len(dictionary.get(token)))
        else:
            idf = 0
        query_dictionary[token] = tf * idf
    return query_dictionary


def query_tf_idf(query_dict, relevant_documents, dictionary, tf_idfs, document_norms):
    results = []

    # calc query vector norm
    query_norm = 0
    for token in query_dict:
        query_norm += query_dict[token] * query_dict[token]
    query_norm = math.sqrt(query_norm)

    for doc in relevant_documents:
        dot = 0
        for token in query_dict:
            if token in dictionary:
                if doc in dictionary[token]:
                    dot += tf_idfs[doc][token] * query_dict[token]

        cosSim = dot / (document_norms[doc] * query_norm)

        results.append((doc, cosSim))

    results.sort(key=lambda t: t[1], reverse=True)
    return results


def query():
    index_path = sys.argv[3]
    method = sys.argv[2]

    # open inverted index and unpack it
    try:
        index_json = open(index_path, "r")
    except:
        print("index path does not exist")
        return

    corpus = json.load(index_json)
    dictionary = corpus["dictionary"]
    document_norms = corpus["document_norms"]
    document_lens = corpus["document_lens"]
    bm25_idfs = corpus["bm25_idfs"]
    tf_idfs = corpus["tf_idfs"]

    index_json.close()

    # preprocess query, tokenize, truncate, and remove stop words
    query = preprocess_query()

    if query is None:
        print("Query question is missing from input.")
        return

    relavent_docs = find_relevant_documents(query, dictionary)
    f = open("ranked_query_docs.txt", "w")
    if method == "bm25":
        results = query_bm_25(query, relavent_docs, dictionary, document_lens, bm25_idfs)
        for i in range(0, len(results)):
            if results[i][1] >= 10:
                f.write(results[i][0] + "\n")
    else:
        query_dict = calc_query_tf_idf(query, dictionary, len(document_lens))

        results = query_tf_idf(query_dict, relavent_docs, dictionary, tf_idfs, document_norms)

        for i in range(0, len(results)):
            if results[i][1] >= 0.075:
                f.write(results[i][0] + "\n")


    f.close()


if __name__ == '__main__':
    if sys.argv[1] == "query":
        query()
    else:
        create_index()
