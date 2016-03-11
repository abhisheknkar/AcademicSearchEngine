__author__ = 'Abhishek'
# NOTE: Python passes objects by value! Make a copy if required
import networkx as nx
import scipy.sparse as sp
from sets import Set

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

def relativeSim(G, maxdist=3):
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

def nodeAtHops(G,start,currhops=0,maxhops=1, visited=[], dir2nodes={}):
    visited2 = visited + [start]

    if currhops <= maxhops or len(G.neighbors(start))==0:
        if currhops > 0:
            if start not in dir2nodes:
                dir2nodes[start] = []
            dir2nodes[start].append(visited2)
        if currhops == maxhops:
            return dir2nodes

    if len(G.neighbors(start))>0:
        for nbr in G.neighbors(start):
            if nbr not in visited2:
                dir2nodes = nodeAtHops(G, nbr, currhops+1, maxhops, visited2, dir2nodes)

    return dir2nodes

if __name__ == "__main__":
    filename = 'data/test.txt'

    G = nx.read_adjlist(filename, create_using=nx.DiGraph())
    S = relativeSim(G,3)
    print S
