__author__ = 'Abhishek'
# NOTE: Python passes objects by value! Make a copy if required
import networkx as nx
import scipy.sparse as sp
from sets import Set
import numpy as np
from CentralitySimilarity import *
import cPickle as pickle
import time

def getValidPapers(minIn=20, minOut=20):
    print "Loading Data..."
    print "Data loaded..." + str(time.time() - time_start) + " seconds."
    print "Getting valid papers..." + str(time.time() - time_start) + " seconds."
    validID = {}
    for key in domainContents:
        for node in domainContents[key]:
            if len(graph.neighbors(node)) >= minIn and len(graph.predecessors(node)) >= minOut:
                if key not in validID:
                    validID[key] = []
                validID[key].append(node)

    return validID

def getSimilaritywrtNode(node1, maxdist=3, alpha=0.5):
    simdict = {}
    reachables = nodeAtHops(graph,node1,0,maxdist,[],{},reverse=True)
    for node2 in reachables:
        # print node1, node2, str(time.time() - time_start) + " seconds."
        sim = 0
        simdict[node2] = sim
    return simdict

if __name__ == "__main__":
    time_start = time.time()
    filename = 'data/test.txt'

    with open('data/domainWiseDBLP.pickle', 'rb') as handle:
        (domaingraph, domainContents, domainabstracts) = pickle.load(handle)
    with open('data/domainWiseDBLP.pickle', 'rb') as handle:
        (graph, domainContents, domainabstracts) = pickle.load(handle)
    # with open('data/allDBLP.pickle', 'rb') as handle:
    #     (graph, contents) = pickle.load(handle)

    validID = getValidPapers()
    SimDict = {}
    for field in validID:
        for paper in validID[field]:
            print "Computing similarity for " + paper
            SimDict[paper] = getSimilaritywrtNode(paper)

    print "Finished in " + str(time.time() - time_start) + " seconds."
