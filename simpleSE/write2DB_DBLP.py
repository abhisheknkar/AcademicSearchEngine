__author__ = 'abhishek'
import networkx as nx
import numpy as np
import MySQLdb as mdb
from os import listdir
import cPickle as pickle
import time

def readDBLPfile(filepath, cur, tablename, domain="", useDomain=False, domainContents={}, graph=nx.DiGraph(), abstracts = {}, doneSet = set()):
    f = open(filepath,'r')
    print domain

    curr_paper = {}
    count = 0

    if useDomain:
        domainContents[domain] = []

    for lineRaw in f.readlines():
        line = lineRaw.strip('\r\n')
        if len(line)>1:
            type = line[1]
        else:
            continue

        if type == '*':
            count += 1
            if count % 1000 == 0:
                print count

            if len(curr_paper) > 0:
                if curr_paper['id'] not in doneSet:
                    cur.execute("INSERT INTO " + tablename + "(id,title,authors,year) VALUES (%s, %s, %s, %s)", (int(curr_paper['id']), curr_paper['title'], curr_paper['authors'], int(curr_paper['year'])))
                    doneSet.add(curr_paper['id'])
                if 'abstract' in curr_paper:
                    if len(curr_paper['abstract']) > 0:
                        abstracts[curr_paper['id']]=curr_paper['abstract']
                if useDomain:
                    domainContents[domain].append(curr_paper['id'])
            curr_paper = {}
            curr_paper['title'] = line[2:]
        elif type == '@':
            # authors = line[2:].split(',')
            curr_paper['authors']= line[2:]
        elif type == 't':
            curr_paper['year']= int(line[2:])
        elif type == 'i':
            curr_paper['id']= line[6:]
        elif type == '%':
            graph.add_edge(curr_paper['id'], line[2:])
        elif type == '!':
            curr_paper['abstract']= line[2:]
    return (graph, domainContents, abstracts, doneSet)

def saveDomainWiseTable():
    saveFlag = True
    con = mdb.connect('localhost', 'abhishek', 'Pass@1234', 'aminerV7');
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS domainWisePapers")
    cur.execute("CREATE TABLE domainWisePapers(id INT PRIMARY KEY AUTO_INCREMENT, title VARCHAR(1000), authors VARCHAR(1000), year INT, FULLTEXT(title)) ENGINE=MyISAM")

    datasetPath = '/home/csd154server/Abhishek_Narwekar/DDP/Datasets/DBLP_citation_2014_May/domains/'
    domainGraph = nx.DiGraph()
    domainContents = {}
    domainabstracts = {}
    doneSet = set()

    files = listdir(datasetPath)
    for file in files:
        if file.endswith('.txt'):
            domain = file[0:-4]
            filePath = datasetPath + file
            (graph, domainContents, domainabstracts, doneSet) = readDBLPfile(filePath, cur, tablename='domainWisePapers', domain=domain, useDomain=True, domainContents=domainContents, graph=domainGraph, abstracts = domainabstracts, doneSet=doneSet)
    with open('data/domainWiseDBLP.pickle', 'wb') as handle:
        print str(time.time() - time_start) + " seconds."
        print "Saving..."
        pickle.dump((domainGraph, domainContents, domainabstracts), handle)

    if saveFlag == False:
        with open('data/domainWiseDBLP.pickle', 'rb') as handle:
            (domainGraph, domainContents, domainabstracts) = pickle.load(handle)

    cur.execute('ALTER TABLE domainWisePapers ADD INDEX (title)')
    con.commit()
    con.close()

def saveAllTable():
    saveFlag = True
    con = mdb.connect('localhost', 'abhishek', 'Pass@1234', 'aminerV7');
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS allPapers")
    cur.execute("CREATE TABLE allPapers(id INT PRIMARY KEY AUTO_INCREMENT, title VARCHAR(1000), authors VARCHAR(1000), year INT, FULLTEXT(title)) ENGINE=MyISAM")

    datasetPath = '/home/csd154server/Abhishek_Narwekar/DDP/Datasets/DBLP_citation_2014_May/'
    graph = nx.DiGraph()
    abstracts = {}
    doneSet = set()

    files = listdir(datasetPath)
    for file in files:
        if file.endswith('.txt'):
            domain = file[0:-4]
            filePath = datasetPath + file
            (graph, domainContents, abstracts, doneSet) = readDBLPfile(filePath, cur, tablename='allPapers', domain=domain, doneSet=doneSet)
    with open('data/allDBLP.pickle', 'wb') as handle:
        print str(time.time() - time_start) + " seconds."
        print "Saving..."
        pickle.dump((graph, abstracts), handle)

    if saveFlag == False:
        with open('data/allDBLP.pickle', 'rb') as handle:
            (domainGraph, domainContents, domainabstracts) = pickle.load(handle)
    cur.execute('ALTER TABLE allPapers ADD INDEX (title)')
    con.commit()
    con.close()

if __name__=="__main__":
    time_start = time.time()
    # saveDomainWiseTable()
    saveAllTable()

    print "Finished in " + str(time.time() - time_start) + " seconds."