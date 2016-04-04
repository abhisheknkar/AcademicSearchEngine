__author__ = 'abhishek'
import networkx as nx
import numpy as np
import scipy as sp
import MySQLdb as mdb
from os import listdir
import cPickle as pickle
import time

def getGenCocitationSim1(graph, alpha=0.5, iter = 5):
    print "Computing Similarity... " + str(time.time() - time_start) + " seconds."

    A = nx.adjacency_matrix(graph)
    I = sp.sparse.identity(A.shape[0])
    # Phi = sp.sparse.linalg.inv((I-alpha*A.T)*(I-alpha*A))
    # Phi = np.linalg.inv((I-alpha*A.T)*(I-alpha*A))
    print "Here!"
    # X = alpha*(A+A.T) + alpha*alpha*A.T*A
    X = sp.sparse.csr_matrix(alpha*(A+A.T), dtype=np.int64)
    Phi = sp.sparse.csr_matrix(I)
    phi = Phi
    A = []
    for i in range(1,iter+1):
        print i, phi.shape
        phi = sp.sparse.csr_matrix.dot(phi, X)
        Phi = Phi + phi
    print "Done!"
    # print "PageRank computed! Saving... " + str(time.time() - time_start) + " seconds."
    # with open('data/PageRank.pickle', 'wb') as handle:
    #     pickle.dump(pr, handle)
    #     print "Done! ..." + str(time.time() - time_start) + " seconds."

def loadgraph(filename):
    print "Loading Data..."
    # with open('data/domainWiseDBLP.pickle', 'rb') as handle:
    #     (graph, domainContents, domainabstracts) = pickle.load(handle)
    with open(filename, 'rb') as handle:
        # (graph, abstracts) = pickle.load(handle)
        (graph, contents, abstracts) = pickle.load(handle)
    return (graph, contents)

if __name__=="__main__":
    time_start = time.time()
    (graph, contents) = loadgraph('data/domainWiseDBLP.pickle')
    getGenCocitationSim1(graph)
    # (graph, contents) = loadgraph('data/allDBLP.pickle')
    # getGenCocitationSim1(graph)

    print "Finished in " + str(time.time() - time_start) + " seconds."