__author__ = 'abhishek'
import networkx as nx
import numpy as np
import MySQLdb as mdb
from os import listdir
import cPickle as pickle
import time

def computePageRank():
    print "Loading Data..."
    # with open('data/domainWiseDBLP.pickle', 'rb') as handle:
    #     (graph, domainContents, domainabstracts) = pickle.load(handle)
    with open('data/allDBLP.pickle', 'rb') as handle:
        (graph, abstracts) = pickle.load(handle)
    print "Computing PageRank... " + str(time.time() - time_start) + " seconds."
    pr = nx.pagerank(graph)
    print "PageRank computed! Saving... " + str(time.time() - time_start) + " seconds."
    with open('data/PageRank.pickle', 'wb') as handle:
        pickle.dump(pr, handle)
        print "Done! ..." + str(time.time() - time_start) + " seconds."

def testFunction():
    print "Loading Data..."
    # with open('data/domainWiseDBLP.pickle', 'rb') as handle:
    #     (graph, domainContents, domainabstracts) = pickle.load(handle)
    with open('data/domainWiseDBLP.pickle', 'rb') as handle:
        # (graph, abstracts) = pickle.load(handle)
        (domainGraph, domainContents, domainabstracts) = pickle.load(handle)
    # print len(graph.nodes())

if __name__=="__main__":
    time_start = time.time()

    # computePageRank()
    testFunction()

    print "Finished in " + str(time.time() - time_start) + " seconds."