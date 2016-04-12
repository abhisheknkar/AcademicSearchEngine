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
    print "XXX"
    genSharedRefs = getGeneralizedCocitation(graph, node1, maxDist, maxDist, reverse=True)
    print "YYY"
    simdictCC = {}
    simdictSR = {}
    for node2 in genCocitations:
        if node2 not in simdictCC:
            simdictCC[node2] = 0
        for path in genCocitations[node2]:
            simdictCC[node2] += alpha**(len(path)-1)
    for node2 in genSharedRefs:
        if node2 not in simdictSR:
            simdictSR[node2] = 0
        for path in genSharedRefs[node2]:
            simdictSR[node2] += alpha**(len(path)-1)
    return (simdictCC, simdictSR)

def getGeneralizedCocitation2(G,start,maxhops1=1, maxhops2=1, alpha=0.5, reverse=False):
    reachables1 = nodeAtHops(G,start,0,maxhops1,[],{},reverse=not(reverse))
    genCocitations = {}
    count = 0
    for node1 in reachables1:
        count += 1
        print str(count) + " out of " + str(len(reachables1))
        reachables2 = nodeAtHops(G,node1,0,maxhops1,[],{},reverse=reverse)
        for node2 in reachables2:
            if node2 not in genCocitations:
                genCocitations[node2] = 0
            if start == node2:
                continue
            paths = []
            for path1 in reachables1[node1]:
                for path2 in reachables2[node2]:
                    finpath = path1 + path2[1:]
                    genCocitations[node2] += alpha**(len(finpath)-1)
    return genCocitations

def getSimilaritywrtNode2(node1, maxDist=3, alpha=0.5):
    simdictCC = getGeneralizedCocitation2(graph, node1, maxDist, maxDist, alpha)
    simdictSR = getGeneralizedCocitation2(graph, node1, maxDist, maxDist, alpha, reverse=True)
    return (simdictCC, simdictSR)

def clusterMetric(saveSimDictFlag=False, topK=50, maxDist=3, simDictLoc='data/SimDict.pickle', p2FMaploc='data/paper2Field.pickle'):
    validID = getValidPapers(minIn=topK, minOut=topK)
    SimDictCC = {}
    SimDictSR = {}
    SimDict = {}

    with open(p2FMaploc, 'rb') as handle:
        paper2FieldMap = pickle.load(handle)
    count = 0
    if saveSimDictFlag == True:
        for field in validID:
            for paper in validID[field]:
                count = count + 1
                print "Computing similarity for " + paper + " " + str(count)
                (SimDictCC[paper], SimDictSR[paper]) = getSimilaritywrtNode2(paper, maxDist=maxDist)
                keySet = set(SimDictCC[paper]).union(set(SimDictSR[paper])).union(set(graph.successors(paper))).union(graph.predecessors(paper))
                SimDict[paper] = dict([(key,0) for key in keySet])
                for key in SimDictCC[paper]:
                    SimDict[paper][key] += SimDictCC[paper][key]
                for key in SimDictSR[paper]:
                    SimDict[paper][key] += SimDictSR[paper][key]
                for key in graph.successors(paper):
                    SimDict[paper][key] += 1
                for key in graph.predecessors(paper):
                    SimDict[paper][key] += 1

            with open(simDictLoc, 'wb') as handle:
                pickle.dump((SimDictCC, SimDictSR, SimDict), handle)

    with open(simDictLoc, 'rb') as handle:
        (SimDictCC, SimDictSR, SimDict) = pickle.load(handle)

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

def computeAvgPrecision(input, till):
    currSum = 0
    precisionVec = []
    for i in range(0,till):
        currSum += input[i]
        precisionVec.append(float(currSum) / (i+1))
    precisionVec = [a*b for a,b in zip(precisionVec,input[0:till])]
    MAP = np.sum(precisionVec) / np.sum(input)
    print precisionVec
    return MAP

def getSimDictforQueries():
    global domaingraph, domainContents, domainabstracts, graph, contents
    topK = 50
    maxDist = 3
    with open('data/domainWiseDBLP.pickle', 'rb') as handle:
        (domaingraph, domainContents, domainabstracts) = pickle.load(handle)
    # with open('data/domainWiseDBLP.pickle', 'rb') as handle:
    #     (graph, domainContents, domainabstracts) = pickle.load(handle)
    with open('data/allDBLP.pickle', 'rb') as handle:
       (graph, contents) = pickle.load(handle)
    pM3 = clusterMetric(saveSimDictFlag=True, topK=topK, maxDist=maxDist, simDictLoc='data/SimDict_' +str(topK) + '_' + str(maxDist) + '.pickle', p2FMaploc='data/paper2Field.pickle')
    pM1 = clusterMetric(saveSimDictFlag=True, topK=topK, maxDist=1, simDictLoc='data/SimDict_' + str(topK) + '_1.pickle', p2FMaploc='data/paper2Field.pickle')
    # pM3 = clusterMetric(saveSimDictFlag=True, topK=topK, maxDist=maxDist, simDictLoc='data/SMALL/SimDict_' +str(topK) + '_' + str(maxDist) + '.pickle', p2FMaploc='data/paper2Field.pickle')
    # pM1 = clusterMetric(saveSimDictFlag=True, topK=topK, maxDist=1, simDictLoc='data/SMALL/SimDict_' + str(topK) + '_1.pickle', p2FMaploc='data/paper2Field.pickle')

    # with open('results/SMALL/Comparison_' + str(topK) + 'vs1.txt', 'wb') as handle:
    with open('results/Comparison_' + str(topK) + 'vs1.txt', 'wb') as handle:
        for key in pM3:
            if key in pM1:
                handle.write(key + '\t' + str(pM3[key]) + '\t' + str(pM1[key]) + '\n')
    handle.close()

if __name__ == "__main__":
    time_start = time.time()



    print "Finished in " + str(time.time() - time_start) + " seconds."
