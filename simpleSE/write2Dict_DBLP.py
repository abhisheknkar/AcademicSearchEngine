__author__ = 'csd154server'
import networkx as nx
import numpy as np
from os import listdir
import pickle
import time

def readDBLPfile(filepath,domain="", useDomain=False, domainContents={}, papers={}, graph=nx.DiGraph()):
    f = open(filePath,'r')
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
                papers[curr_paper['id']] = curr_paper
                if useDomain:
                    domainContents[domain].append(curr_paper['id'])
            curr_paper = {}
            curr_paper['title'] = line[2:]
        elif type == '@':
            authors = line[2:].split(',')
            curr_paper['authors']= authors
        elif type == 't':
            curr_paper['year']= int(line[2:])
        elif type == 'i':
            curr_paper['id']= line[6:]
        elif type == '%':
            graph.add_edge(curr_paper['id'], line[2:])
        elif type == '!':
            curr_paper['abstract']= line[2:]
    return (papers, graph, domainContents)


if __name__=="__main__":
    time_start = time.time()
    save = False
    # outFile = 'domainWiseDBLP.pickle'
    outFile = 'allDBLP.pickle'

    if save:
        # datasetPath = '/home/csd154server/Abhishek_Narwekar/DDP/Datasets/DBLP_citation_2014_May/domains/'
        datasetPath = '/home/csd154server/Abhishek_Narwekar/DDP/Datasets/DBLP_citation_2014_May/'
        domainGraph = nx.DiGraph()
        domainContents = {}
        domainPapers = {}

        files = listdir(datasetPath)
        for file in files:
            if file.endswith('.txt'):
                domain = file[0:-4]
                filePath = datasetPath + file
                # (domainPapers, domainGraph, domainContents) = readDBLPfile(filePath,domain,useDomain=True, domainContents=domainContents, papers=domainPapers, graph=domainGraph)
                (papers, graph, domainContents) = readDBLPfile(filePath)

        with open(outFile, 'wb') as handle:
            print str(time.time() - time_start) + " seconds."
            print "Saving..."
            # pickle.dump((domainPapers, domainGraph, domainContents), handle)
            pickle.dump((papers, graph), handle)
    else:
        # with open('domainWiseDBLP.pickle', 'rb') as handle:
        #     (domainPapers, domainGraph, domainContents) = pickle.load(handle)
        with open('allDBLP.pickle', 'rb') as handle:
            (papers, graph) = pickle.load(handle)

    print "Finished in " + str(time.time() - time_start) + " seconds."