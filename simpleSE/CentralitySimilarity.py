__author__ = 'Abhishek'
# NOTE: Python passes objects by value! Make a copy if required
import networkx as nx
import scipy.sparse as sp
from sets import Set
import numpy as np

def DFSpaths(G, curr, dest, visited, paths):
    visited2 = visited + [curr]
    if curr==dest:
        paths.append(visited2)

    else:
        if len(G.neighbors(curr))>0:
            for nbr in G.neighbors(curr):
                if nbr not in visited2:
                    paths = DFSpaths(G, nbr, dest, visited2, paths)
    return paths

def computeSimilarityAll(G, maxdist=3):
    S = {}
    for node in G.nodes():
        S[node] = {}    #Initialize the similarity matrix
    reachables = {}

    for k in G.nodes():
        reachables[k] = nodeAtHops(G,k,0,maxdist,[],{}) #Get all paths reachable from all nodes of length less than max

    for node in G.nodes():
        done = Set()
        for dest1 in reachables[node]:
            done.add(dest1) #To skip repetitions
            for dest2 in reachables[node]:
                if dest2 in done:
                    continue    #To skip repetitions
                for path1 in reachables[node][dest1]:
                    for path2 in reachables[node][dest2]:   #All path pairs
                        if len(Set(path1).intersection(Set(path2))) == 1:    #Disjoint
                            if dest2 not in S[dest1]:
                                S[dest1][dest2] = 0
                                S[dest2][dest1] = 0
                            toadd = 1.0 / ( len(path1) + len(path2) - 2)   #1 / length added
                            S[dest1][dest2] += toadd
                            S[dest2][dest1] += toadd
    return S

def computeSimilarityPair(G, node1, node2, maxdist, alpha, disjoint=False):
    reachables1 = nodeAtHops(G,node1,0,maxdist,[],{},reverse=True) #Get all paths reachable from node1 of length less than max
    reachables2 = nodeAtHops(G,node2,0,maxdist,[],{},reverse=True) #Get all paths reachable from node2 of length less than max

    commonDest = set(reachables1.keys()).intersection(set(reachables2.keys()))

    sim12 = 0
    for dest in commonDest:
        for path1 in reachables1[dest]:
            for path2 in reachables2[dest]:
                if disjoint==True:
                    if len(set(path1[1:-1]).intersection(set(path2[1:-1]))) > 0:
                        continue
                totalLength = len(path1) + len(path2) - 2
                # print path1, path2
                sim12 += len2simFunc(totalLength, alpha)
    return sim12

def len2simFunc(length, alpha=0.5):
    return alpha**length

def nodeAtHops(G,start,currhops=0,maxhops=1, visited=[], dir2nodes={}, reverse=False):
    visited2 = visited + [start]

    if currhops <= maxhops or len(G.neighbors(start))==0:
        if currhops > 0:
            if start not in dir2nodes:
                dir2nodes[start] = []
            dir2nodes[start].append(visited2)
        if currhops == maxhops:
            return dir2nodes

    if reverse == False:
        nbrs = G.neighbors(start)
    else:
        nbrs = G.predecessors(start)

    if len(nbrs)>0:
        for nbr in nbrs:
            if nbr not in visited2:
                dir2nodes = nodeAtHops(G, nbr, currhops+1, maxhops, visited2, dir2nodes, reverse=reverse)

    return dir2nodes

def getGeneralizedCocitation(G,start,maxhops1=1, maxhops2=1, reverse=False):
    reachables1 = nodeAtHops(G,start,0,maxhops1,[],{},reverse=not(reverse))
    genCocitations = {}
    for node1 in reachables1:
        reachables2 = nodeAtHops(G,node1,0,maxhops1,[],{},reverse=reverse)
        for node2 in reachables2:
            if start == node2:
                continue
            paths = []
            for path1 in reachables1[node1]:
                for path2 in reachables2[node2]:
                    paths.append(path1 + path2[1:])
            genCocitations[node2] = paths
    return genCocitations

if __name__ == "__main__":
    filename = '../data/test.txt'

    G = nx.read_adjlist(filename, create_using=nx.DiGraph())
    S = getGeneralizedCocitation(G,u'1', 3, 3, reverse=False)
    print S

    # S = computeSimilarityPair(G,node1=u'4',node2=u'5',maxdist=3, disjoint=True)
    # print S