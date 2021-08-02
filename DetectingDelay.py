import networkx as nx
import sys
from gurobipy import *
import math
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
import numpy as np
import time
import random


def DetectDelay(graphs, S, Nodes):
    expected_delay = 1000
    #looping through each simulation
    for graph in graphs:
        #starting from any of the lists of infected nodes
        for ListOfInfectedNodes in Nodes:
            for node in ListOfInfectedNodes:
                V_id = nx.shortest_path_length(G, node)
                #if any of our monitored nodes are found within a delay shorter than our current shortest delay lower the expected delay
                for target in S:
                    for key in V_id.keys()
                        if target in V_id[key] and key < expected_delay:
                            expected_delay = key
                            break #Since we loop over the keys from 0 to x, once we find a node at d distance away, that is our minimum from the given start node so no need to check more
    return expected_delay
