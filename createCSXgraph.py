__author__ = 'csd154server'
import networkx as nx
import pickle
import time

def readCSXgraph(filename):
    f = open(filename,'r')
    G = nx.DiGraph()
    count = 0

    for line in f.readlines():
        count+=1
        if count % 10000 == 0:
            print str(count) + " lines read."

        if line[-2]==':':
            source = line[0:-2]
        else:
            dest = line[0:-1]
            G.add_edge(source,dest)
    f.close()
    return G

if __name__=="__main__":
    start_time = time.time()

    # G = readCSXgraph('/home/csd154server/Abhishek_Narwekar/DDP/Datasets/CiteSeerX/data/citegraph.txt')
    # with open('/home/csd154server/Abhishek_Narwekar/DDP/Datasets/CiteSeerX/data/CSXgraph.pickle', 'wb') as handle:
    #     pickle.dump(G, handle)

    with open('/home/csd154server/Abhishek_Narwekar/DDP/Datasets/CiteSeerX/data/CSXgraph.pickle', 'rb') as handle:
        G = pickle.load(handle)

    print len(G.edges())

    print("--- %s seconds ---" % (time.time() - start_time))