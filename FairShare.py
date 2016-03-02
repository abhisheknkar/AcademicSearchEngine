__author__ = 'Abhishek'
import pprint
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

def getD(G, alpha=0.9, gamma = 1):
    # This scheme is way too strict
    beta = 1 - alpha
    global A,V,n,VInd

    D = np.array([[[0 for k in xrange(n)] for j in xrange(n)] for i in xrange(n)],dtype=np.float) #Initialize

    # Setting various elements
    for node in VInd:
        # Get neighbours of that node
        nbrs = G.neighbors(node)
        nonnbrs = list(set(V) - set(nbrs) - set(node))

        cocitlist = []
        noncocitlist = []
        for nbr in nbrs:
            for nbr2 in G.neighbors(nbr):
                if G.has_edge(node,nbr2):
                    cocitlist.append((nbr,nbr2))
                else:
                    noncocitlist.append((nbr,nbr2))
        # print node,cocitlist,noncocitlist

        toSet1 = toSet2 = 0
        if len(cocitlist) > 0:
            toSet1 = alpha / (len(cocitlist))    # For the co-citations
        if len(noncocitlist) > 0:
            toSet2 = beta / (len(noncocitlist)) # For the non-co-citations

        # for nbr in nbrs:
        #     D[VInd[node],VInd[node],VInd[nbr]] = 1
        for cocit in cocitlist:
            D[VInd[node],VInd[cocit[0]],VInd[cocit[1]]] = toSet1
        for noncocit in noncocitlist:
            D[VInd[node],VInd[noncocit[0]],VInd[noncocit[1]]] = toSet2

    # D[i,i,k]
    for node1 in VInd:
        D_all = sum(sum(D[:,VInd[node1],:]))
        if D_all > 0:
            for node2 in VInd:
                D[VInd[node1],VInd[node1],VInd[node2]] = sum(D[:,VInd[node1],VInd[node2]]) / D_all

    pprint.pprint(D)
    return D

def getC(G, D, iterations=1):
    eps = np.finfo(float).eps
    global A,V,n,VInd

    C = sp.sparse.lil_matrix(A / np.tile(sp.sparse.csr_matrix.sum(A,axis=1)+eps,(1,n))) #Initialize

    for iter in range(iterations):
        for j in range(n):
            for k in range(n):
                if j != k:
                    C[j,k] = C[j,k] + C[:,j].T.dot(D[:,j,k])
    return C

def fairShareMain(G,iter=1):
    global A,V,n,VInd

    # Global variables
    V = G.nodes()
    n = len(V)
    A = nx.adjacency_matrix(G, nodelist=V)
    VInd = {}
    for i,elem in enumerate(V):
        VInd[elem] = i

    D = getD(G)
    C = getC(G, D, iterations=iter)

    Csum = sp.sparse.lil_matrix.sum(C,axis=0)
    CsumNorm = Csum / np.sum(Csum)

    print V
    print CsumNorm

def comparisons(G):
    evec = nx.eigenvector_centrality(G)
    print "EVEC:",evec

    pagerank = nx.pagerank(G)
    print "PAGERANK: ", pagerank

    katz = nx.katz_centrality(G)
    print "KATZ: ", katz

if __name__ == "__main__":
    filename = 'data/tenpol2.txt'

    G = nx.read_adjlist(filename, create_using=nx.DiGraph())
    fairShareMain(G,iter=1)
    # comparisons(G)

    # nx.draw_networkx(G)
    # plt.show()
