__author__ = 'abhishek'

import pydot

graph = pydot.Dot(graph_type='digraph')

f = open('Q2.txt','r')

for line in f.readlines():
    linesplit = line.split()
    edge = pydot.Edge(linesplit[0], linesplit[1])
    graph.add_edge(edge)

graph.write_ps('Q2.ps')
graph.write_svg('Q2.svg')
