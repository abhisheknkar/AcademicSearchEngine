__author__ = 'abhishek'

import pydot

graph = pydot.Dot(graph_type='digraph')

f = open('data/test.txt','r')

for line in f.readlines():
    linesplit = line.split()
    edge = pydot.Edge(linesplit[0], linesplit[1])
    graph.add_edge(edge)

# graph.write_ps('test.ps')
graph.write_svg('test.svg')
