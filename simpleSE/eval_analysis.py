__author__ = 'abhishek'
import networkx as nx
import scipy.sparse as sp
from sets import Set
import numpy as np
from CentralitySimilarity import *
import cPickle as pickle
import time
import operator

with open('data/allDBLP.pickle', 'rb') as handle:
   (graph, contents) = pickle.load(handle)
topK = 50
maxDist = 3
simDictLoc='data/SimDict_' +str(topK) + '_' + str(maxDist) + '.pickle'
type = 'new'
# type = 'baseline'
with open('data/paper2Field.pickle', 'rb') as handle:
    paper2FieldMap = pickle.load(handle)

with open(simDictLoc, 'rb') as handle:
    (SimDictCC, SimDictSR, SimDict) = pickle.load(handle)

SortedSimDict = {}
percentageMatch = {}
for key in SimDict:
    # f = open("results/queryWise/" + str(topK) + "/" + key + ".txt", "wb")
    f = open("results/queryWise/" + type + '/'+ 'ALL' + "/" + key + ".txt", "wb")
    f.write("Current paper: " + key + "; Cluster: " + paper2FieldMap[key] + '\n')
    print "Current paper: " + key + "; Cluster: " + paper2FieldMap[key]
    SortedSimDict[key] = sorted(SimDict[key].items(), key=operator.itemgetter(1),reverse=True)
    currCluster = paper2FieldMap[key]
    matches = 0
    refLocs = []
    citLocs = []
    count = 0
    for i in range(0,len(SimDict[key])):
        count += 1
        toPrint = SortedSimDict[key][i][0]
        if SortedSimDict[key][i][0] in paper2FieldMap:
            if paper2FieldMap[SortedSimDict[key][i][0]] == currCluster:
                toPrint += "\tSameCluster"
                matches += 1
            else:
                toPrint += "\tDiffCluster"
        else:
            toPrint += "\tNoCluster"
        if SortedSimDict[key][i][0] in graph.successors(key):
            toPrint += "\tReference"
            refLocs.append(i)
        elif SortedSimDict[key][i][0] in graph.predecessors(key):
            toPrint += "\tCitation"
            citLocs.append(i)
        else:
            toPrint += "\tNoLink\t"
        toPrint += "\t" + str(SortedSimDict[key][i][1])
        f.write(toPrint + "\n")
    percentageMatch[key] = float(matches) / count
    f.write("Percentage Match = " + str(percentageMatch[key]) + '\n')
    f.write("Reference Locs: " + str(refLocs) +'\n')
    f.write("Citation Locs: " + str(citLocs) + '\n')
    f.close()