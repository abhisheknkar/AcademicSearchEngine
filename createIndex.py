__author__ = 'abhishek'
from os import listdir

def addToRevIndex(file, index):
    with open (file, "r") as myfile:
        text=myfile.read()
    text = cleanText(text)
    words = text.split()
    for word in words:
        if word not in index:
            index[word] = {}
        if file in index[word]:
            index[word][file] += 1
        else:
            index[word][file] = 1
    return index

def cleanText(text):
    text.replace("."," ")
    text.replace(",", " ")
    return text

def getCandidateDocs(query, index):
    candidateDocs = set()
    for word in query.split():
        if word in index:
            candidateDocs = candidateDocs.union(set(index[word].keys()))
    return candidateDocs

# def rankDocs(candidateDocs, G):

if __name__ == "__main__":
    folder = '/home/abhishek/Datasets/bbcsport/cricket/'
    query = "we want answers"
    revIndex = {}
    for file in listdir(folder):
        revIndex = addToRevIndex(folder + file, revIndex)
    candidateDocs = getCandidateDocs(query, revIndex)

