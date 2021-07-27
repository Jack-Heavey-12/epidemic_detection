import networkx as nx
import sys
from gurobipy import *
import math
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
import numpy as np
import time
import random

from operator import itemgetter

def RandomSolution(graph, k):
    S = []
    for node in graph.nodes:
        if len(S) <= k:
            p = random.uniform(0,1)
            if p > 0.25:
                S.append(node)
        else:
            break
    return S

def DegreeSolution(graph, k):
    S = []
    SortedNodes = sorted(graph.degree_iter(),key=itemgetter(1),reverse=True)
    for i in range(0, k):
        S.append(SortedNodes[i])
    return S

def CentralitySolution(graph, k):
    S = []
    centrality = nx.eigenvector_centrality(graph)
    CentralNodes = []
    for node in centrality.keys():
        pair = (node, centrality[node])
        CentralNodes.append(pair)
        sorted(CentralNodes, key=lambda x: x[1])
    for i in range(0, k):
        S.append(CentralNodes[i])
    return S 

def PageRankSolution(graph, k):
    S = []
    PageRank = nx.pagerank(G)
    PRNodes = []
    for node in PageRank.keys():
        pair = (node, PageRank[node])
        PRNodes.append(pair)
        sorted(PRNodes, key=lambda x: x[1])
    for i in range(0, k):
        S.append(PRNodes[i])
    return S

