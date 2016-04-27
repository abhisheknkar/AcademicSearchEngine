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
import random
import shutil

def computeAvgPrecision(input, till):
    currSum = 0
    precisionVec = []
    for i in range(0,till):
        currSum += input[i]
        precisionVec.append(float(currSum) / (i+1))
    precisionVec = [a*b for a,b in zip(precisionVec,input[0:till])]
    MAP = np.sum(precisionVec) / np.sum(input)
    return MAP

def getValidPapersUnsegregated(minIn=20, minOut=20):
    validID = []
    for node in graph:
        if len(graph.neighbors(node)) >= minIn and len(graph.predecessors(node)) >= minOut:
            validID.append(node)
    return validID

def scoreGeneralizedCocitations(G,start,maxhops1=1, maxhops2=1, alpha=0.5, reverse=False, traditionalFlag=False):
    reachables1 = nodeAtHops(G,start,0,maxhops1,[],{},reverse=not(reverse))
    genCocitations = {}
    count = 0
    for node1 in reachables1:
        count += 1
        reachables2 = nodeAtHops(G,node1,0,maxhops1,[],{},reverse=reverse)
        for node2 in reachables2:
            if start == node2:
                continue
            paths = []
            for path1 in reachables1[node1]:
                for path2 in reachables2[node2]:
                    finpath = path1 + path2[1:]
                    # if len(finpath)-1>4:
                    #     continue
                    if node2 not in genCocitations:
                        genCocitations[node2] = 0
                    if traditionalFlag:
                        genCocitations[node2] += 1
                    else:
                        genCocitations[node2] += alpha**(len(finpath)-1)
    return genCocitations

def saveSimDictforQuery(graph, paper, outputFolder, topK=100, maxDist=3, alpha=0.5, SMALLflag=True, traditionalFlag=False):
    outputPath = outputFolder + '/' + paper +'.pkl'

    count = 0

    simDictCC = scoreGeneralizedCocitations(graph, paper, maxDist, maxDist, alpha, traditionalFlag=traditionalFlag)
    simDictSR = scoreGeneralizedCocitations(graph, paper, maxDist, maxDist, alpha, reverse=True, traditionalFlag=traditionalFlag)
    with open(outputPath, 'wb') as handle:
        pickle.dump((simDictCC, simDictSR), handle)

def mergeSimDictCCSR(SimDictCC, SimDictSR, graph, paper):
    keySet = set(SimDictCC).union(set(SimDictSR)).union(set(graph.successors(paper))).union(graph.predecessors(paper))
    SimDict = dict([(key,0) for key in keySet])
    for key in SimDictCC[paper]:
        SimDict[key] += SimDictCC[key]
    for key in SimDictSR[paper]:
        SimDict[key] += SimDictSR[key]
    for key in graph.successors(paper):
        SimDict[key] += 1
    for key in graph.predecessors(paper):
        SimDict[key] += 1
    return SimDict

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


    # print "Computing average precision..."
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
            # print linesplit, col
            if linesplit[col] in relevant:
                MAPdict[file].append(1)
            if linesplit[col] in ignore:
                continue
            else:
                # linesplit[1] == irrelevant:
                MAPdict[file].append(0)
        f.close()

    # print "Computing average precision..."
    f = open(outfile, 'w')
    for key in MAPdict:
        MAPvals[key] = computeAvgPrecision(MAPdict[key], till=len(MAPdict[key]))
        f.write(key + "\t" + str(MAPvals[key]) + "\n")
    f.close()
    return MAPdict

def callMAPRefPred(filepath, outfile, col, relevant, ignore, maxlines=1000):
    MAPdict = {}
    MAPvals = {}
    # print "Parsing file:"
    f = open(filepath,'r')
    count = 0
    MAPdict[filepath] = []
    for line in f.readlines():
        count += 1
        if count == 1:
            continue
        if count > maxlines:
            break
        linesplit = line.split()
        # print linesplit, col
        # print linesplit[col], linesplit[col] in relevant
        if linesplit[col] in relevant:
            MAPdict[filepath].append(1)
        if linesplit[col] in ignore:
            continue
        else:
            # linesplit[1] == irrelevant:
            MAPdict[filepath].append(0)
    f.close()

    # print "Computing average precision..."
    f = open(outfile, 'a')
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
        # print "Current paper: " + key + "; Cluster: " + paper2FieldMap[key]
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

def getAverageDistanceFromQuery(simDict, graph, source, limitTo=100):
    graphUD = graph.to_undirected()
    sortedSimDict = sorted(simDict.items(), key=operator.itemgetter(1),reverse=True)

    topKsimNodes = zip(*sortedSimDict[0:min(limitTo,len(sortedSimDict))])[0]
    distances = []
    for dest in topKsimNodes:
        distances.append(len(nx.shortest_path(graphUD, source, dest)) - 1)
    return np.mean(distances)

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
            if removeRef in graphMod[paper]:
                graphMod.remove_edge(paper, removeRef)
            else:
                print "NE"

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
            topK, maxDist, type=type, smallflag=True, limitTo=1000)

    foldername = 'results/refpred/' + str(alpha) + '/trial' + str(trial) + '/' + type + '/'
    outFile = 'results/refpred/' + str(alpha) + '/trial' + str(trial) + '/' + type + '.txt'
    relevant = set([refs[x] for x in toRemove])
    # print relevant

    # for id in relevant:
    #     print paper, id, str(computeSimilarityPair(graphMod, paper, id, maxDist, alpha, disjoint=False))

    rankedListPath = 'results/refpred/' + str(alpha) + '/trial' + str(trial) + '/' + type + '/' + paper + '.txt'

    callMAPRefPred(rankedListPath, outFile, 0, relevant, [], maxlines=100) #***

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
    # print "YOYO: " + paper + '; Length: ' + str(len(SimDict))
    return SimDict

def callReferencePrediction(topK, maxDist, alpha, predictPercent=20, trial=0):
    con = mdb.connect('localhost', 'abhishek', 'Pass@1234', 'aminerV7');
    loadGraphs(smallflag=True)
    validID = getValidPapers(minIn=topK, minOut=topK)

    if os.path.exists('results/refpred/' + str(alpha) + '/trial' + str(trial)):
        shutil.rmtree('results/refpred/' + str(alpha) + '/trial' + str(trial))

    if not os.path.exists('results/refpred/' + str(alpha) + '/trial' + str(trial) + '/simDicts/'):
        # os.makedirs('results/refpred/' + str(alpha) + '/trial' + str(trial))
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
            # toRemove = [randint(0,len(refs)-1) for x in range(0,int((predictPercent*len(refs)/100)))]
            toRemove = random.sample(range(0,len(refs)-1),int((predictPercent*len(refs)/100)))
            referencePrediction(topK, maxDist, alpha, predictPercent, con, paper, trial, toRemove)
            referencePrediction(topK, 1, alpha, predictPercent, con, paper, trial, toRemove)

def ReferencePredictionMain():
    for alpha in [0.1,0.3,0.5,0.7,0.9]:
        for trial in range(0,20):
            print "Current iteration: ", alpha, trial
            callReferencePrediction(topK=30, maxDist=3, alpha=alpha, predictPercent=20, trial=trial)

def ReferencePredictionAggregation():
    files = [0,0]
    rawscores = {'baseline':{}, 'new':{}}
    zeroCount = {'baseline':{}, 'new':{}}
    types = ['baseline', 'new']
    summaryscores = {'baseline':{}, 'new':{}}
    alphas = [0.1,0.3,0.5,0.7,0.9]
    for alpha in alphas:
        for trial in range(0,20):
            files[0] = 'results/refpred/' + str(alpha) + '/trial' + str(trial) + '/baseline.txt'
            files[1] = 'results/refpred/' + str(alpha) + '/trial' + str(trial) + '/new.txt'
            for idx,file in enumerate(files):
                f = open(file,'r')
                for line in f.readlines():
                    linesplit = line.split()
                    paper = linesplit[0][linesplit[0].rfind('/')+1:-4]
                    if paper not in zeroCount[types[idx]]:
                        zeroCount[types[idx]][paper] = 0
                    if paper not in rawscores[types[idx]]:
                        rawscores[types[idx]][paper] = {}
                    if alpha not in rawscores[types[idx]][paper]:
                        rawscores[types[idx]][paper][alpha] = []
                    if linesplit[1] == 'nan':
                        score = 0
                        zeroCount[types[idx]][paper] += 1
                    else:
                        score = float(linesplit[1])

                    rawscores[types[idx]][paper][alpha].append(score)
                f.close()

    for type in types:
        for paper in rawscores[type]:
            summaryscores[type][paper] = []
            # print paper
            for alpha in alphas:
                # summaryscores[paper].append((np.mean(rawscores[paper][alpha]), np.std(rawscores[paper][alpha])))
                summaryscores[type][paper].append(np.mean(rawscores[type][paper][alpha]))
            # print summaryscores[type][paper]

    f = open('results/refpred/summary.txt','w')
    for paper in rawscores['baseline']:
        f.write("Paper: " + paper + '\n')
        f.write("Baseline: " + str(summaryscores['baseline'][paper]) + "\n")
        f.write("New: " + str(summaryscores['new'][paper]) + "\n\n")

    f.write("ZeroCount: " + str(zeroCount))

def getSimPathsInFile(maxDist=3, minLinks=5, smallflag=True, numberOfQueries=10):
    loadGraphs(smallflag)
    if smallflag:
        folder='data/simPaths/SMALL/maxDist='+str(maxDist) + '/'
    else:
        folder='data/simPaths/BIG/maxDist='+str(maxDist) + '/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    validIDs = getValidPapersUnsegregated(minLinks,minLinks)
    allValidPaps = []
    queryIDs = random.sample(range(0,len(validIDs)-1), numberOfQueries)
    for idx, queryID in enumerate(queryIDs):
        query = validIDs[queryID]
        print idx, query
        writeGeneralizedCocitation(graph,folder,query,maxDist, maxDist, reverse=False)
        writeGeneralizedCocitation(graph,folder,query,maxDist, maxDist, reverse=True)
    # getSimPathsInFile(maxDist=3, minLinks=5, smallflag=True, numberOfQueries=50)

def getDistancesForRandomQueries(folderPrefix, outFile, writeflag=True, readflag=False, summaryFile='results/task2Summary.txt'):
    alphaList = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 'traditional']
    # alphaList = ['traditional']
    if writeflag:
        loadGraphs(smallflag=True)

        DistancesDict = {'CC':{},'SR':{},'merged':{}}
        if os.path.exists(outFile):
            with open(outFile, 'r') as handle:
                DistancesDict = pickle.load(handle)

        for alpha in alphaList:
            DistancesDict['CC'][str(alpha)] = {}
            DistancesDict['SR'][str(alpha)] = {}
            files = os.listdir(folderPrefix+'alpha='+str(alpha))
            for file in files:
                source = file[:-4]
                print alpha, source
                filepath = folderPrefix + 'alpha='+ str(alpha) + '/' + file
                with open(filepath, 'rb') as handle:
                    (simDictCC,simDictSR) = pickle.load(handle)
                DistancesDict['CC'][str(alpha)][source] = getAverageDistanceFromQuery(simDictCC, graph, source, limitTo=100)
                DistancesDict['SR'][str(alpha)][source] = getAverageDistanceFromQuery(simDictSR, graph, source, limitTo=100)

        with open(outFile, 'wb') as handle:
            pickle.dump(DistancesDict, handle)
    if readflag:
        # READ THE OUPUTS AND WRITE TO A FILE
        with open(outFile, 'r') as handle:
            DistancesDict = pickle.load(handle)

        f = open(summaryFile, 'w')
        f.write('Alpha\tCC\t\SR\n')
        for alpha in alphaList:
            CCs = np.array(DistancesDict['CC'][str(alpha)].values())
            CCs[np.isnan(CCs)]=0
            SRs = np.array(DistancesDict['SR'][str(alpha)].values())
            SRs[np.isnan(SRs)]=0

            CCmean = np.mean(CCs)
            SRmean = np.mean(SRs)

            f.write(str(alpha) + '\t' + str(CCmean) + '\t' + str(SRmean) + '\n')

def task2CallSequence():
    # saveSimDictForRandomQueries(smallflag=True, numberOfQueries = 50, minLinks = 5, outputFolderPrefix='/home/csd154server/Abhishek_Narwekar/DDP/ASONAM/simDictSmall/')
    getDistancesForRandomQueries(folderPrefix='/home/csd154server/Abhishek_Narwekar/DDP/ASONAM/simDictSmall/', outFile='results/avgdistances.pkl', writeflag=False, readflag=True, summaryFile='results/task2Summary.txt')

def task1CallSequence():
    getClusterScoresForRandomQueries(folderPrefix='/home/csd154server/Abhishek_Narwekar/DDP/ASONAM/simDictSmall/', outFile='results/ClusterMAPs.pkl',  writeflag=False, readflag=True, summaryFile='results/task1Summary.txt')

def getClusterScoresForRandomQueries(folderPrefix='/home/csd154server/Abhishek_Narwekar/DDP/ASONAM/simDictSmall/', outFile='results/ClusterMAPs.pkl',  writeflag=True, readflag=True, summaryFile='results/task1Summary.txt'):
    alphaList = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 'traditional']
    # alphaList = ['traditional']

    with open('data/paper2Field.pickle', 'r') as handle:
        paper2FieldMap = pickle.load(handle)

    if writeflag:
        loadGraphs(smallflag=True)

        ClusterMAPsGeneralized = {'CC':{},'SR':{},'merged':{}}
        if os.path.exists(outFile):
            with open(outFile, 'r') as handle:
                ClusterMAPsGeneralized = pickle.load(handle)

        for alpha in alphaList:
            ClusterMAPsGeneralized['CC'][str(alpha)] = {}
            ClusterMAPsGeneralized['SR'][str(alpha)] = {}
            files = os.listdir(folderPrefix+'alpha='+str(alpha))
            for file in files:
                source = file[:-4]
                print alpha, source
                filepath = folderPrefix + 'alpha='+ str(alpha) + '/' + file
                with open(filepath, 'rb') as handle:
                    (simDictCC,simDictSR) = pickle.load(handle)
                ClusterMAPsGeneralized['CC'][str(alpha)][source] = getClusterMAPScoreforSimDict(simDictCC, source, paper2FieldMap, topK=100)
                ClusterMAPsGeneralized['SR'][str(alpha)][source] = getClusterMAPScoreforSimDict(simDictSR, source, paper2FieldMap, topK=100)

        with open(outFile, 'wb') as handle:
            pickle.dump(ClusterMAPsGeneralized, handle)

    if readflag:
        # READ THE OUPUTS AND WRITE TO A FILE
        with open(outFile, 'r') as handle:
            ClusterMAPsGeneralized = pickle.load(handle)

        f = open(summaryFile, 'w')
        f.write('Alpha\tCC\t\SR\n')
        for alpha in alphaList:
            CCs = np.array(ClusterMAPsGeneralized['CC'][str(alpha)].values())
            CCs[np.isnan(CCs)]=0
            SRs = np.array(ClusterMAPsGeneralized['SR'][str(alpha)].values())
            SRs[np.isnan(SRs)]=0

            CCmean = np.mean(CCs)
            SRmean = np.mean(SRs)
            f.write(str(alpha) + '\t' + str(CCmean) + '\t' + str(SRmean) + '\n')

def getClusterMAPScoreforSimDict(SimDict, currPaper, paper2FieldMap, topK):
    SortedSimDict = sorted(SimDict.items(), key=operator.itemgetter(1),reverse=True)
    currCluster = paper2FieldMap[currPaper]
    relevanceVec = []
    for i in range(0,min(topK, len(SortedSimDict))):
        if SortedSimDict[i][0] in paper2FieldMap:
            if paper2FieldMap[SortedSimDict[i][0]] == currCluster:
                relevanceVec.append(1)
            else:
                relevanceVec.append(0)
        else:
            relevanceVec.append(0)
    return computeAvgPrecision(relevanceVec, min(topK, len(SortedSimDict)))

def saveSimDictForRandomQueries(smallflag=True, numberOfQueries = 10, minLinks = 5, outputFolderPrefix='data/simDictSmall/'):
    loadGraphs(smallflag=True)
    alphaList = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

    # if os.path.exists(outputFolderPrefix):
    #     shutil.rmtree(outputFolderPrefix)

    validIDs = getValidPapersUnsegregated(minLinks,minLinks)
    queryIDs = random.sample(range(0,len(validIDs)-1), numberOfQueries)
    for idx, queryID in enumerate(queryIDs):
        print "Saving SimDict for: ", idx, validIDs[queryID]
        query = validIDs[queryID]
        for alpha in alphaList:
            print 'alpha = ', alpha
            outputFolder = outputFolderPrefix + 'alpha=' + str(alpha) + '/'
            if not os.path.exists(outputFolder):
                os.makedirs(outputFolder)
            saveSimDictforQuery(graph, query, outputFolder, maxDist=3, alpha=alpha, SMALLflag=True, traditionalFlag=False)

        print 'Traditional'
        outputFolder = outputFolderPrefix + 'traditional/'
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        saveSimDictforQuery(graph, query, outputFolder, maxDist=3, SMALLflag=True, traditionalFlag=True)

if __name__ == "__main__":
    time_start = time.time()
    task2CallSequence()

    print "Finished in " + str(time.time() - time_start) + " seconds."


# CALL sequence:
# getSimDictforQueries() #Gets similarity dicts for all valid papers
# SimDict2RankedList() #Gets a ranked list corresponding to a similarity dictionary
# callMAP() #Analyzes the ranked list and gets average precision
# compareMAPS() #Compares 2 maps