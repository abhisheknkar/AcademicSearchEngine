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
import os
import MySQLdb as mdb
from random import randint

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
        # print str(count) + " out of " + str(len(reachables1))
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
                    if len(finpath)-1>4:
                        continue
                    genCocitations[node2] += alpha**(len(finpath)-1)
    return genCocitations

def getSimilaritywrtNode2(graph, node1, maxDist=3, alpha=0.5):
    simdictCC = getGeneralizedCocitation2(graph, node1, maxDist, maxDist, alpha)
    simdictSR = getGeneralizedCocitation2(graph, node1, maxDist, maxDist, alpha, reverse=True)
    return (simdictCC, simdictSR)

def clusterMetric(graph, saveSimDictFlag=False, alpha=0.5, topK=50, maxDist=3, simDictLoc='data/SimDict.pickle', p2FMaploc='data/paper2Field.pickle'):
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
                (SimDictCC[paper], SimDictSR[paper]) = getSimilaritywrtNode2(graph, paper, maxDist=maxDist)
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
    return MAP

def getSimDictforQueries(topK=20, maxDist=3, alpha=0.5, SMALLflag=True):
    global domaingraph, domainContents, domainabstracts, graph, contents
    print topK, maxDist, alpha, SMALLflag
    with open('data/domainWiseDBLP.pickle', 'rb') as handle:
        (domaingraph, domainContents, domainabstracts) = pickle.load(handle)
    if SMALLflag == True:
        with open('data/domainWiseDBLP.pickle', 'rb') as handle:
            (graph, domainContents, domainabstracts) = pickle.load(handle)
        pM3 = clusterMetric(graph, saveSimDictFlag=True, alpha=alpha, topK=topK, maxDist=maxDist, simDictLoc='data/SMALL/SimDict_' +str(topK) + '_' + str(maxDist) + '_' + str(alpha) + '.pickle', p2FMaploc='data/paper2Field.pickle')
        pM1 = clusterMetric(graph, saveSimDictFlag=True, alpha=alpha, topK=topK, maxDist=1, simDictLoc='data/SMALL/SimDict_' + str(topK) + '_1_' + str(alpha) + '.pickle', p2FMaploc='data/paper2Field.pickle')
        compFileName = 'results/queryWise/SMALL/Comparison_' + str(topK) + 'vs1_' + str(alpha) + '.txt'
    else:
        with open('data/allDBLP.pickle', 'rb') as handle:
           (graph, contents) = pickle.load(handle)
        pM1 = clusterMetric(graph, saveSimDictFlag=True, alpha=alpha, topK=topK, maxDist=1, simDictLoc='data/SimDict_' + str(topK) + '_1_' + str(alpha) + '.pickle', p2FMaploc='data/paper2Field.pickle')
        pM3 = clusterMetric(graph, saveSimDictFlag=True, alpha=alpha, topK=topK, maxDist=maxDist, simDictLoc='data/SimDict_' +str(topK) + '_' + str(maxDist) + '_' + str(alpha) + '.pickle', p2FMaploc='data/paper2Field.pickle')
        compFileName = 'results/queryWise/BIG/Comparison_' + str(topK) + 'vs1_' + str(alpha) + '.txt'

    # with open(compFileName, 'wb') as handle:
    #     for key in pM3:
    #         if key in pM1:
    #             handle.write(key + '\t' + str(pM3[key]) + '\t' + str(pM1[key]) + '\n')
    # handle.close()
    # getSimDictforQueries()

def callMAPCluster(foldername, outfile, maxlines=1000):
    files = os.listdir(foldername)
    MAPdict = {}
    MAPvals = {}
    print "Parsing file:"
    for file in files:
        print file
        filepath = foldername + file
        f = open(filepath,'r')
        count = 0
        MAPdict[file] = []
        for line in f.readlines():
            count += 1
            if count == 1:
                continue
            if count > maxlines:
                break
            linesplit = line.split()
            if linesplit[1] == 'SameCluster':
                MAPdict[file].append(1)
            if linesplit[1] == 'DiffCluster':
                MAPdict[file].append(0)
        f.close()


    print "Computing average precision..."
    f = open(outfile, 'w')
    for key in MAPdict:
        MAPvals[key] = computeAvgPrecision(MAPdict[key], till=len(MAPdict[key]))
        f.write(key + "\t" + str(MAPvals[key]) + "\n")
    f.close()
    return MAPdict
    # callMAP('results/queryWise/new/ALL/', 'results/MAP/new.txt')
    # callMAP('results/queryWise/baseline/ALL/', 'results/MAP/baseline.txt')
    # callMAP('results/SMALL/queryWise/baseline/ALL/', 'results/MAP/SMALL/100/baseline.txt', maxlines=100)
    # callMAP('results/SMALL/queryWise/new/ALL/', 'results/MAP/SMALL/100/new.txt', maxlines=100)

def callMAP(foldername, outfile, col, relevant, ignore, maxlines=1000):
    files = os.listdir(foldername)
    MAPdict = {}
    MAPvals = {}
    print "Parsing file:"
    for file in files:
        print file
        filepath = foldername + file
        f = open(filepath,'r')
        count = 0
        MAPdict[file] = []
        for line in f.readlines():
            count += 1
            if count == 1:
                continue
            if count > maxlines:
                break
            linesplit = line.split()
            print linesplit, col
            if linesplit[col] in relevant:
                MAPdict[file].append(1)
            if linesplit[col] in ignore:
                continue
            else:
                # linesplit[1] == irrelevant:
                MAPdict[file].append(0)
        f.close()

    print "Computing average precision..."
    f = open(outfile, 'w')
    for key in MAPdict:
        MAPvals[key] = computeAvgPrecision(MAPdict[key], till=len(MAPdict[key]))
        f.write(key + "\t" + str(MAPvals[key]) + "\n")
    f.close()
    return MAPdict


def compareMAPS(foldername):
    f1 = open(foldername + '/new.txt','r')
    f2 = open(foldername + '/baseline.txt','r')
    l1 = []
    l2 = []
    for line in f1.readlines():
        linesplit = line.split()
        l1.append(float(linesplit[1]))
    f1.close()
    for line in f2.readlines():
        linesplit = line.split()
        l2.append(float(linesplit[1]))
    f2.close()

    f3 = open(foldername + '/comparison.txt','wb')
    f3.write('New mean: ' + str(np.mean(l1)) + '; New SD: ' + str(np.std(l1)) + '\n')
    f3.write('Baseline mean: ' + str(np.mean(l2)) + '; Baseline SD: ' + str(np.std(l2)) + '\n')
    f3.close()
    # compareMAPS('results/MAP')
    # compareMAPS('results/MAP/SMALL/100')

def SimDict2RankedList(simDictLoc, outputFolder, topK=20, maxDist=3, type='new', smallflag=True, limitTo=1000):
    # Types: new or baseline
    if smallflag == True:
        # simDictLoc='data/SMALL/SimDict_' +str(topK) + '_' + str(maxDist) + '.pickle'
        with open('data/domainWiseDBLP.pickle', 'rb') as handle:
            (graph, domainContents, domainabstracts) = pickle.load(handle)
    else:
        with open('data/allDBLP.pickle', 'rb') as handle:
           (graph, contents) = pickle.load(handle)
        # simDictLoc='data/SimDict_' +str(topK) + '_' + str(maxDist) + '.pickle'
    with open('data/paper2Field.pickle', 'rb') as handle:
        paper2FieldMap = pickle.load(handle)

    with open(simDictLoc, 'rb') as handle:
        (SimDictCC, SimDictSR, SimDict) = pickle.load(handle)

    SortedSimDict = {}
    percentageMatch = {}
    for key in SimDict:
        # if smallflag == True:
            # if not os.path.exists("results/SMALL/queryWise/" + type + '/'+ 'ALL' + "/"):
        if not os.path.exists(outputFolder + type + '/'):
            os.makedirs(outputFolder + type + '/')
        f = open(outputFolder + type + '/'+ key + ".txt", "wb")
        # else:
        #     if not os.path.exists("results/queryWise/" + type + '/'+ 'ALL' + "/"):
        #         os.makedirs("results/queryWise/" + type + '/'+ 'ALL' + "/")
        #     f = open("results/queryWise/" + type + '/'+ 'ALL' + "/" + key + ".txt", "wb")
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
            if count > limitTo:
                break
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
        # percentageMatch[key] = float(matches) / count
        # f.write("Percentage Match = " + str(percentageMatch[key]) + '\n')
        # f.write("Reference Locs: " + str(refLocs) +'\n')
        # f.write("Citation Locs: " + str(citLocs) + '\n')
        f.close()
    # SimDict2RankedList(topK=20, maxDist=3, type='new', smallflag=True)
    # SimDict2RankedList(topK=50, maxDist=1, type='baseline', smallflag=False)

def getDistancefromQuery(foldername, outFolder, smallflag, limitTo=1000):
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    graphUD = graph.to_undirected()

    files = os.listdir(foldername)
    distances = {}
    allList = []
    fout = open(outFolder + '/avgDistances.txt', 'wb')
    for file in files:
        if not file.endswith('.txt'):
            continue
        print "Processing file: " + file

        source = file[0:-4]
        filepath = foldername + '/' + file
        f = open(filepath, 'r')
        count = 0
        distances[file] = {}
        for line in f.readlines():
            count += 1
            if count % 100 == 0:
                print "Completed nodes: " + str(count)
            if count == 1:
                continue
            if count > limitTo + 1:
                break
            linesplit = line.split()
            if len(linesplit) != 4:
                continue
            dest = linesplit[0]
            if source in graph and dest in graph:
                distances[file][dest] = len(nx.shortest_path(graphUD, source, dest)) - 1
        avgDist = np.mean(distances[file].values())
        stdDist = np.std(distances[file].values())
        allList += distances[file].values()
        fout.write('Source: ' + source + '; Mean Dist: ' + str(avgDist) + '; Std: ' + str(stdDist) + '\n')

        f.close()
    fout.write('Overall - ' + 'Mean Dist: ' + str(np.mean(allList)) + '; Std: ' + str(np.std(allList)) + '\n')
    fout.close()
    with open(outFolder + '/distances.pickle', 'wb') as handle:
        pickle.dump(distances, handle)
    # getDistancefromQuery(foldername='results/queryWise/SMALL/new/ALL', outFolder='results/distances/SMALL/new/', smallflag=True, limitTo=1000)
    # getDistancefromQuery(foldername='results/queryWise/SMALL/baseline/ALL', outFolder='results/distances/SMALL/baseline/', smallflag=True, limitTo=1000)
    # getDistancefromQuery(foldername='results/queryWise/BIG/baseline/ALL', outFolder='results/distances/BIG/baseline/', smallflag=False, limitTo=1000)
    # getDistancefromQuery(foldername='results/queryWise/BIG/new/ALL', outFolder='results/distances/BIG/new/', smallflag=False, limitTo=1000)

def loadGraphs(smallflag):
    global graph, contents, domainContents, domainabstracts
    if smallflag == True:
        with open('data/domainWiseDBLP.pickle', 'rb') as handle:
            (graph, domainContents, domainabstracts) = pickle.load(handle)
    else:
        with open('data/allDBLP.pickle', 'rb') as handle:
           (graph, contents) = pickle.load(handle)
    print "Graph loaded"

def referencePrediction(topK, maxDist, alpha, predictPercent, con, paper, trial, toRemove):
    cur = con.cursor()

    graphMod = graph.copy()
    refs = graphMod.successors(paper)
    cur.execute("SELECT * FROM domainWisePapers WHERE ID = " + paper)
    paperInfo = cur.fetchall()[0]
    currYear = paperInfo[3]

    cur.execute("SELECT * FROM domainWisePapers WHERE year>" + str(currYear))
    recentPapers = cur.fetchall()

    for index in toRemove:
        removeRef = refs[index]
        if removeRef in graphMod:
            graphMod.remove_node(removeRef)

    for recentPaper in recentPapers:
        recentID = str(recentPaper[0])
        if recentID in graphMod:
            graphMod.remove_node(recentID)
    # clusterMetric(graphMod, saveSimDictFlag=True, alpha=alpha, topK=topK, maxDist=maxDist, simDictLoc='results/refpred/' + str(alpha) + '/simDicts/SimDict_' +str(topK) + '_' + str(maxDist) + '_' + paper + '.pickle', p2FMaploc='data/paper2Field.pickle')
    simDictLoc='results/refpred/' + str(alpha) + '/trial' + str(trial) + '/simDicts/SimDict_' +str(topK) + '_' + str(maxDist) + '_' + paper + '.pickle'
    SimDictCC = {}
    SimDictSR = {}
    SimDict = {}
    (SimDictCC[paper], SimDictSR[paper]) = getSimilaritywrtNode2(graphMod, paper, maxDist, alpha)
    SimDict[paper] = mergeSimDictsCCSR(SimDictCC[paper], SimDictSR[paper], graphMod, paper)

    with open(simDictLoc, 'wb') as handle:
        pickle.dump((SimDictCC, SimDictSR, SimDict), handle)

    if maxDist == 1:
        type = 'baseline'
    else:
        type = 'new'
    SimDict2RankedList(simDictLoc, 'results/refpred/' + str(alpha) + '/trial' + str(trial) + '/',
            topK, maxDist, type=type, smallflag=True, limitTo=len(refs))

    foldername = 'results/refpred/' + str(alpha) + '/trial' + str(trial) + '/' + type + '/'
    outFile = foldername + type + '.txt'
    relevant = set([refs[x] for x in toRemove])
    callMAP(foldername, outFile, 2, relevant, [], maxlines=1000)

def mergeSimDictsCCSR(SimDictCC, SimDictSR, graph, paper):
    SimDict = {}
    keySet = set(SimDictCC).union(set(SimDictSR)).union(set(graph.successors(paper))).union(graph.predecessors(paper))
    SimDict = dict([(key,0) for key in keySet])
    for key in SimDictCC:
        SimDict[key] += SimDictCC[key]
    for key in SimDictSR:
        SimDict[key] += SimDictSR[key]
    for key in graph.successors(paper):
        SimDict[key] += 1
    for key in graph.predecessors(paper):
        SimDict[key] += 1
    return SimDict

def callReferencePrediction(topK, maxDist, alpha, predictPercent=20, trial=0):
    con = mdb.connect('localhost', 'abhishek', 'Pass@1234', 'aminerV7');
    loadGraphs(smallflag=True)
    validID = getValidPapers(minIn=topK, minOut=topK)

    if not os.path.exists('results/refpred/' + str(alpha) + '/trial' + str(trial)):
        os.makedirs('results/refpred/' + str(alpha) + '/trial' + str(trial))
        os.makedirs('results/refpred/' + str(alpha) + '/trial' + str(trial) + '/simDicts/')
    f = open('results/refpred/' + str(alpha) + '/trial' + str(trial) + '/params.txt', 'wb')
    f.write('topK=' + str(topK) + '\nmaxDist=' + str(maxDist) + '\nalpha=' + str(alpha) + '\npredictPercent=' + str(predictPercent))
    f.close()

    count = 0
    for field in validID:
        for paper in validID[field]:
            count += 1
            print "Getting similarity for: " + paper + "; count = " + str(count)
            refs = graph.successors(paper)
            toRemove = [randint(0,len(refs)-1) for x in range(0,int((predictPercent*len(refs)/100)))]
            referencePrediction(topK, maxDist, alpha, predictPercent, con, paper, trial, toRemove)
            referencePrediction(topK, 1, alpha, predictPercent, con, paper, trial, toRemove)

if __name__ == "__main__":
    time_start = time.time()

    # callReferencePrediction(topK=50, maxDist=3, alpha=0.5, trial=0)
    getSimDictforQueries(topK=20, maxDist=3, alpha=0.1, SMALLflag=True)
    getSimDictforQueries(topK=20, maxDist=3, alpha=0.3, SMALLflag=True)
    getSimDictforQueries(topK=20, maxDist=3, alpha=0.7, SMALLflag=True)
    getSimDictforQueries(topK=20, maxDist=3, alpha=0.9, SMALLflag=True)

    getSimDictforQueries(topK=50, maxDist=3, alpha=0.3, SMALLflag=False)
    getSimDictforQueries(topK=50, maxDist=3, alpha=0.5, SMALLflag=False)
    getSimDictforQueries(topK=50, maxDist=3, alpha=0.7, SMALLflag=False)

    print "Finished in " + str(time.time() - time_start) + " seconds."


# CALL sequence:
# getSimDictforQueries() #Gets similarity dicts for all valid papers
# SimDict2RankedList() #Gets a ranked list corresponding to a similarity dictionary
# callMAP() #Analyzes the ranked list and gets average precision
# compareMAPS() #Compares 2 maps

    # TO RUN:

    # getSimDictforQueries(topK=50, maxDist=3, alpha=0.1, SMALLflag=False)
    # getSimDictforQueries(topK=50, maxDist=3, alpha=0.3, SMALLflag=False)
    # getSimDictforQueries(topK=50, maxDist=3, alpha=0.5, SMALLflag=False)
    # getSimDictforQueries(topK=50, maxDist=3, alpha=0.7, SMALLflag=False)

    # getSimDictforQueries(topK=20, maxDist=3, alpha=0.1, SMALLflag=True)
    # getSimDictforQueries(topK=20, maxDist=3, alpha=0.3, SMALLflag=True)
    # getSimDictforQueries(topK=20, maxDist=3, alpha=0.7, SMALLflag=True)
    # getSimDictforQueries(topK=20, maxDist=3, alpha=0.9, SMALLflag=True)
