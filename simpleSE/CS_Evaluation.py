__author__ = 'Abhishek'
# NOTE: Python passes objects by value! Make a copy if required
import networkx as nx
import scipy.sparse as sp
from sets import Set
import numpy as np
from CentralitySimilarity import *
import cPickle as pickle
import time
import operator

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

def getSimilaritywrtNode(node1, maxDist=3, alpha=0.5):
    genCocitations = getGeneralizedCocitation(graph, node1, maxDist, maxDist)
    simdict = {}
    for node2 in genCocitations:
        if node2 not in simdict:
            simdict[node2] = 0
        for path in genCocitations[node2]:
            simdict[node2] += alpha**(len(path)-1)
    return simdict

def clusterMetric(saveSimDictFlag=False, topK=50, maxDist=3, simDictLoc='data/SimDict.pickle', p2FMaploc='data/paper2Field.pickle'):
    validID = getValidPapers(minIn=topK, minOut=topK)
    SimDict = {}

    with open(p2FMaploc, 'rb') as handle:
        paper2FieldMap = pickle.load(handle)
    count = 0
    if saveSimDictFlag == True:
        for field in validID:
            for paper in validID[field]:
                count = count + 1
                print "Computing similarity for " + paper + " " + str(count)
                SimDict[paper] = getSimilaritywrtNode(paper, maxDist=maxDist)
        with open(simDictLoc, 'wb') as handle:
            pickle.dump(SimDict, handle)

    with open(simDictLoc, 'rb') as handle:
        SimDict = pickle.load(handle)
    SortedSimDict = {}
    percentageMatch = {}
    for key in SimDict:
        SortedSimDict[key] = sorted(SimDict[key].items(), key=operator.itemgetter(1),reverse=True)
        currCluster = paper2FieldMap[key]
        matches = 0
        for i in range(0,topK):
            if SortedSimDict[key][i][0] in paper2FieldMap:
                if paper2FieldMap[SortedSimDict[key][i][0]] == currCluster:
                    matches += 1
        percentageMatch[key] = float(matches) / topK

        for key in percentageMatch:
            print key, percentageMatch[key]
    return percentageMatch

if __name__ == "__main__":
    time_start = time.time()
    topK = 40
    maxDist = 3
    with open('data/domainWiseDBLP.pickle', 'rb') as handle:
        (domaingraph, domainContents, domainabstracts) = pickle.load(handle)
    # with open('data/domainWiseDBLP.pickle', 'rb') as handle:
    #     (graph, domainContents, domainabstracts) = pickle.load(handle)
    with open('data/allDBLP.pickle', 'rb') as handle:
       (graph, contents) = pickle.load(handle)
    pM3 = clusterMetric(saveSimDictFlag=True, topK=topK, maxDist=maxDist, simDictLoc='data/SimDict_' +str(topK) + '_' + str(maxDist) + '.pickle', p2FMaploc='data/paper2Field.pickle')
    pM1 = clusterMetric(saveSimDictFlag=False, topK=topK, maxDist=1, simDictLoc='data/SimDict_' + str(topK) + '_1.pickle', p2FMaploc='data/paper2Field.pickle')

    with open('results/Comparison_' + str(topK) + 'vs1.txt', 'wb') as handle:
        for key in pM3:
            if key in pM1:
                handle.write(key + '\t' + str(pM3[key]) + '\t' + str(pM1[key]) + '\n')
    handle.close()

    print "Finished in " + str(time.time() - time_start) + " seconds."
