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
import matplotlib.pyplot as plt

# -----------------------------------------------MASTER FUNCTIONS--------------------------------------------------
def task1CallSequence():
    global finpathDist
    finpathDist = np.zeros(6)
    writeflag=True
    readflag=True
    folderPrefix='/home/csd154server/Abhishek_Narwekar/DDP/ASONAM/simDictSmall/'
    outFile='results/ClusterMAPs.pkl'
    summaryFile='results/task1Summary.txt'
    method = 'VIP'

    saveSimDictForRandomQueries(smallflag=True, numberOfQueries = 1, minLinks = 5, outputFolderPrefix=folderPrefix, method=method)
    getClusterScoresForRandomQueries(folderPrefix=folderPrefix, outFile=outFile,  writeflag=writeflag, readflag=readflag, summaryFile=summaryFile)
    # with open('results/finPathDict.pkl','wb') as handle:
    #     pickle.dump(finpathDist, handle)
    #     print finpathDist

def task2CallSequence():
    # saveSimDictForRandomQueries(smallflag=True, numberOfQueries = 50, minLinks = 5, outputFolderPrefix='/home/csd154server/Abhishek_Narwekar/DDP/ASONAM/simDictSmall/')
    writeflag=True
    readflag=True
    folderPrefix='/home/csd154server/Abhishek_Narwekar/DDP/ASONAM/simDictSmall_VIP/'
    outFile='results/VIP/avgdistances.pkl'
    summaryFile='results/VIP/task2Summary.txt'
    getDistancesForRandomQueries(folderPrefix=folderPrefix, outFile=outFile, writeflag=writeflag, readflag=readflag, summaryFile=summaryFile)

def task3CallSequence():
    doReferencePredictionForRandomQueries()

# -----------------------------------------------GENERAL-----------------------------------------------------------
def loadGraphs(smallflag):
    global graph, contents, domainContents, domainabstracts
    if smallflag == True:
        with open('data/domainWiseDBLP.pickle', 'rb') as handle:
            (graph, domainContents, domainabstracts) = pickle.load(handle)
    else:
        with open('data/allDBLP.pickle', 'rb') as handle:
           (graph, contents) = pickle.load(handle)
    print "Graph loaded"

def computeAvgPrecision(input, till):
    currSum = 0
    precisionVec = []
    for i in range(0,till):
        currSum += input[i]
        precisionVec.append(float(currSum) / (i+1))
    precisionVec = [a*b for a,b in zip(precisionVec,input[0:till])]
    if np.sum(input) == 0:
        MAP = 0
    else:
        MAP = float(np.sum(precisionVec)) / np.sum(input)
    return MAP

def getValidPapersUnsegregated(minIn=20, minOut=20):
    loadGraphs(smallflag=True)
    validID = []
    for node in graph:
        if len(graph.neighbors(node)) >= minIn and len(graph.predecessors(node)) >= minOut:
            validID.append(node)
    return validID

def scoreGeneralizedCocitations(G,start,maxhops1=1, maxhops2=1, alpha=0.5, reverse=False, traditionalFlag=False, method='VDP'):
    if traditionalFlag:
        maxhops1 = 1
        maxhops2 = 1
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
                    if method=='VIP':
                        if len(Set(path1).intersection(Set(path2))) != 1: #Not disjoint
                            break
                    finpath = path1 + path2[1:]
                    # print finpath
                    # if len(finpath)-1>4:
                    #     continue
                    finpathDist[len(finpath)-2] += 1
                    if node2 not in genCocitations:
                        genCocitations[node2] = 0
                    if traditionalFlag:
                        genCocitations[node2] += 1
                    else:
                        genCocitations[node2] += alpha**(len(finpath)-1)
    return genCocitations

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

def getSimDictforQuery(graph, paper, outputFolder='', maxDist=3, alpha=0.5, SMALLflag=True, traditionalFlag=False, saveFlag=False, method='VDP'):
    outputPath = outputFolder + '/' + paper +'.pkl'

    count = 0

    simDictCC = scoreGeneralizedCocitations(graph, paper, maxDist, maxDist, alpha, traditionalFlag=traditionalFlag, method=method)
    simDictSR = scoreGeneralizedCocitations(graph, paper, maxDist, maxDist, alpha, reverse=True, traditionalFlag=traditionalFlag, method=method)

    if saveFlag:
        with open(outputPath, 'wb') as handle:
            pickle.dump((simDictCC, simDictSR), handle)

    return (simDictCC,simDictSR)

def saveSimDictForRandomQueries(smallflag=True, numberOfQueries = 10, minLinks = 5, outputFolderPrefix='data/simDictSmall/', method='VDP'):
    loadGraphs(smallflag=True)
    alphaList = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    # alphaList = [0.01]

    # if os.path.exists(outputFolderPrefix):
    #     shutil.rmtree(outputFolderPrefix)

    # Hardcoding the papers:
    # validIDsFull = os.listdir('/home/csd154server/Abhishek_Narwekar/DDP/ASONAM/simDictSmall/alpha=0.01/')
    # validIDs = [x[:-4] for x in validIDsFull]
    # OR:
    queryIDs = range(0,numberOfQueries)
    validIDs = ['1046227', '907663','1280018', '907669', '995660', '985077', '1010043', '797271', '1014794', '857701', '1023663', '1023252', '861010', '2866181', '860795', '1022869', '1343340', '857669', '1027807', '2998173', '2850836', '907796', '907569', '1270712', '880978', '1141582', '907881', '1045138', '499010', '796576', '835602', '1042957', '1343315', '1027785', '1041452', '570347', '1042555', '831732', '499381', '1042538', '1111869', '907609', '857704', '871766', '2867377', '918302', '2998182', '636918', '985045', '801187']

    # validIDs = getValidPapersUnsegregated(minLinks,minLinks)
    # queryIDs = random.sample(range(0,len(validIDs)-1), numberOfQueries)
    for idx, queryID in enumerate(queryIDs):
        print "Saving SimDict for: ", idx, validIDs[queryID]
        query = validIDs[queryID]
        for alpha in alphaList:
            print 'alpha = ', alpha
            outputFolder = outputFolderPrefix + 'alpha=' + str(alpha) + '/'
            if not os.path.exists(outputFolder):
                os.makedirs(outputFolder)
            getSimDictforQuery(graph, query, outputFolder, maxDist=3, alpha=alpha, SMALLflag=True, traditionalFlag=False, method=method)

        print 'Traditional'
        outputFolder = outputFolderPrefix + 'alpha=traditional/'
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        getSimDictforQuery(graph, query, outputFolder, maxDist=1, SMALLflag=True, traditionalFlag=True, method=method)

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

# -----------------------------------------------EVALUATION 1-----------------------------------------------------------
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
        f.write('Alpha\tCC\tSR\n')
        for alpha in alphaList:
            CCs = np.array(ClusterMAPsGeneralized['CC'][str(alpha)].values())
            CCs[np.isnan(CCs)]=0
            SRs = np.array(ClusterMAPsGeneralized['SR'][str(alpha)].values())
            SRs[np.isnan(SRs)]=0

            CCmean = np.mean(CCs)
            SRmean = np.mean(SRs)
            f.write(str(alpha) + '\t' + str(CCmean) + '\t' + str(SRmean) + '\n')

# -----------------------------------------------EVALUATION 2-----------------------------------------------------------
def getAverageDistanceforSimDict(simDict, graph, source, limitTo=100):
    graphUD = graph.to_undirected()
    sortedSimDict = sorted(simDict.items(), key=operator.itemgetter(1),reverse=True)

    topKsimNodes = zip(*sortedSimDict[0:min(limitTo,len(sortedSimDict))])[0]
    distances = []
    for dest in topKsimNodes:
        distances.append(len(nx.shortest_path(graphUD, source, dest)) - 1)
    return np.mean(distances)

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
                DistancesDict['CC'][str(alpha)][source] = getAverageDistanceforSimDict(simDictCC, graph, source, limitTo=100)
                DistancesDict['SR'][str(alpha)][source] = getAverageDistanceforSimDict(simDictSR, graph, source, limitTo=100)

        with open(outFile, 'wb') as handle:
            pickle.dump(DistancesDict, handle)
    if readflag:
        # READ THE OUPUTS AND WRITE TO A FILE
        with open(outFile, 'r') as handle:
            DistancesDict = pickle.load(handle)

        f = open(summaryFile, 'w')
        f.write('Alpha\tCC\tSR\n')
        for alpha in alphaList:
            CCs = np.array(DistancesDict['CC'][str(alpha)].values())
            CCs[np.isnan(CCs)]=0
            SRs = np.array(DistancesDict['SR'][str(alpha)].values())
            SRs[np.isnan(SRs)]=0

            CCmean = np.mean(CCs)
            SRmean = np.mean(SRs)

            f.write(str(alpha) + '\t' + str(CCmean) + '\t' + str(SRmean) + '\n')

# -----------------------------------------------EVALUATION 3-----------------------------------------------------------
def getMAPforReferencePredictionFromSimDict(SimDict, refs, topK=100):
    SortedSimDict = sorted(SimDict.items(), key=operator.itemgetter(1),reverse=True)

    # print refs
    # print SimDict

    relevanceVec = []
    for i in range(0,min(topK, len(SortedSimDict))):
        if SortedSimDict[i][0] in refs:
            relevanceVec.append(1)
        else:
            relevanceVec.append(0)
    toReturn = computeAvgPrecision(relevanceVec, min(topK, len(SortedSimDict)))
    # print toReturn
    return toReturn

def doReferencePredictionForQuery(paper, con, predictPercent=20, iterations=10, topK=100, maxDist=3, alpha=0.5, SMALLflag=True, traditionalFlag=False, method='VDP'):
    MAP_CC = []
    MAP_SR = []

    if str(alpha) == 'traditional':
        maxDist = 1
    # loadGraphs(smallflag=SMALLflag)
    cur = con.cursor()
    graphMod0 = graph.copy()
    refs = graphMod0.successors(paper)
    cur.execute("SELECT * FROM domainWisePapers WHERE ID = " + paper)
    paperInfo = cur.fetchall()[0]
    currYear = paperInfo[3]

    # Get the papers written after query paper
    cur.execute("SELECT * FROM domainWisePapers WHERE year>" + str(currYear))
    recentPapers = cur.fetchall()
    for recentPaper in recentPapers:
        recentID = str(recentPaper[0])
        if recentID in graphMod0:
            graphMod0.remove_node(recentID)

    # Random node deletion
    for i in range(0,iterations):
        print str(i) + "\t" + str(time.time() - time_start) + " s."
        graphMod = graphMod0.copy()
        toRemove = random.sample(range(0,len(refs)-1),int((predictPercent*len(refs)/100)))

        for index in toRemove:
            removeRef = refs[index]
            if removeRef in graphMod:
                if removeRef in graphMod[paper]:
                    graphMod.remove_edge(paper, removeRef)
                else:
                    print "REFERENCE DOES NOT EXIST"

        relevant = set([refs[x] for x in toRemove])
        # Get the simDicts
        (SimDictCC, SimDictSR) = getSimDictforQuery(graphMod, paper, maxDist=maxDist, alpha=alpha, SMALLflag=True, traditionalFlag=traditionalFlag, saveFlag=False, method=method)

        # Get MAP
        MAP_CC.append(getMAPforReferencePredictionFromSimDict(SimDictCC, relevant, topK=100))
        MAP_SR.append(getMAPforReferencePredictionFromSimDict(SimDictSR, relevant, topK=100))
    return (MAP_CC, MAP_SR)

def doReferencePredictionForRandomQueries():
    writeflag = False
    readflag = True
    outFile = 'results/VIP/refPredMAP.pkl'
    summaryFile = 'results/VIP/task3Summary.txt'
    method = 'VDP'

    con = mdb.connect('localhost', 'abhishek', 'Pass@1234', 'aminerV7')
    alphaList = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 'traditional']
    numberOfQueries=50
    predictPercent=20
    iterations=10
    topK=100
    maxDist=3
    SMALLflag=True
    bigRefPaps = getValidPapersUnsegregated(minIn=5, minOut=10)

    validIDs = ['1046227', '907663','1280018', '907669', '995660', '985077', '1010043', '797271', '1014794', '857701', '1023663', '1023252', '861010', '2866181', '860795', '1022869', '1343340', '857669', '1027807', '2998173', '2850836', '907796', '907569', '1270712', '880978', '1141582', '907881', '1045138', '499010', '796576', '835602', '1042957', '1343315', '1027785', '1041452', '570347', '1042555', '831732', '499381', '1042538', '1111869', '907609', '857704', '871766', '2867377', '918302', '2998182', '636918', '985045', '801187']
    # validIDs = ['907669']
    queryIDs = range(0,numberOfQueries)

    if writeflag:
        loadGraphs(smallflag=True)
        MAPRefPredDict = {'CC':{},'SR':{},'merged':{}}
        if os.path.exists(outFile):
            with open(outFile, 'r') as handle:
                MAPRefPredDict = pickle.load(handle)

        for alpha in alphaList:
            if str(alpha) == 'traditional':
                traditionalFlag=True
            else:
                traditionalFlag=False

            MAPRefPredDict['CC'][str(alpha)] = {}
            MAPRefPredDict['SR'][str(alpha)] = {}

            for queryID in queryIDs:
                paper = validIDs[queryID]
                print alpha, paper, queryID
                (MAPRefPredDict['CC'][str(alpha)][paper], MAPRefPredDict['SR'][str(alpha)][paper]) = doReferencePredictionForQuery(paper, con, predictPercent=predictPercent, iterations=iterations, topK=topK, maxDist=maxDist, alpha=alpha, SMALLflag=True, traditionalFlag=traditionalFlag, method=method)
            with open(outFile, 'wb') as handle:
                pickle.dump(MAPRefPredDict, handle)
    if readflag:
        # READ THE OUPUTS AND WRITE TO A FILE
        with open(outFile, 'r') as handle:
            MAPRefPredDict = pickle.load(handle)

        f = open(summaryFile, 'w')
        f.write('Alpha\tCC\tSR\n')
        for alpha in alphaList:
            CCmeanVec = []
            SRmeanVec = []
            for paper in MAPRefPredDict['CC'][str(alpha)]:
                if paper not in bigRefPaps:
                    continue
                CCraw = np.array(MAPRefPredDict['CC'][str(alpha)][paper])
                CCraw[np.isnan(CCraw)]=0
                CCmeanVec.append(np.mean(CCraw))

                SRraw = np.array(MAPRefPredDict['SR'][str(alpha)][paper])
                SRraw[np.isnan(CCraw)]=0
                SRmeanVec.append(np.mean(SRraw))
            print SRmeanVec
            CCmeanmean = np.mean(CCmeanVec)
            SRmeanmean = np.mean(SRmeanVec)
            CCstdmean = np.std(CCmeanVec)
            SRstdmean = np.std(SRmeanVec)

            f.write(str(alpha) + '\t' + str(CCmeanmean) + '\t' + str(CCstdmean) + '\t' + str(SRmeanmean)+ '\t' + str(SRstdmean) + '\n')
# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------THE REAL DEAL-----------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------

def misc_PathLengthEvaluation():
    VDPvec = np.array([1935,685760,263504,532024,59136],dtype=float)
    VIPvec = np.array([418,85247,31042,62288,2106],dtype=float)
    # print np.sum(VDPvec)
    # print np.sum(VIPvec)
    VDPvec = VDPvec / np.sum(VDPvec) * 100
    VIPvec = VIPvec / np.sum(VIPvec) * 100
    # print VDPvec
    # print VIPvec
    ind = np.arange(5)
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, VDPvec, width, color='r')
    rects2 = ax.bar(ind + width, VIPvec, width, color='y')
    ax.set_ylabel('Percentage of Paths')
    ax.set_title('Path Length')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('2', '3', '4', '5', '6'))
    ax.legend((rects1[0], rects2[0]), ('Generalized', 'Vertex Disjoint'))

    # plt.show()
    plt.savefig("percentComparison.eps", format='eps', dpi=1000)


if __name__ == "__main__":
    global time_start
    time_start = time.time()
    # task1CallSequence()
    # task2CallSequence()
    task3CallSequence()
    # misc_PathLengthEvaluation()



    print "Finished in " + str(time.time() - time_start) + " seconds."