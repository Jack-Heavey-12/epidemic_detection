from GetSolutions import *
import pandas as pd
from typing import List
import networkx as nx
import sys
from gurobipy import *
import math
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
import numpy as np
import time
import random
from collections import defaultdict

#Function for randomly selecting k infection sources for a given sample graph
def RandomInfectionSources(graph, k):
    InfectedNodes = []
    for node in graph.nodes():
        if len(InfectedNodes) < k:
            p = random.uniform(0,1)
            if p > 0.1:
                #print(node)
                InfectedNodes.append(node)
        else:
            break
    return InfectedNodes

#Function for calculating the expected delay for the system
#The function takes as input a simulation graph, a set of nodes of interest S, and a list of infection sources
def DetectDelay(graph, S, Nodes):
    expected_delay = 100000
    #checking distance from any of the given infection sources
    for node in Nodes:
        V_id = nx.shortest_path_length(graph, node)
        #checking if any of our monitored nodes are found within a delay shorter than our current shortest delay lower the expected delay
        for key in list(V_id.keys()):
            if key in S and V_id[key] < expected_delay:
                expected_delay = V_id[key]
                break #Since we loop over the keys from 0 to x, once we find a node at d distance away, that is our minimum from the given start node so no need to check more
    return expected_delay


#Function for sampling edges with probability p. Will return a list of sample graphs.
def sampling(num_samples, graph, p):

	lst = []

	#Generates one graph per sample
	for i in range(num_samples):
		TempGraph = nx.Graph()

		TempGraph.add_nodes_from(graph.nodes())

		#Adds an edge if it is randomly selected with probability p
		for j in graph.edges():
			r = np.random.random()

			if r <= p:
				TempGraph.add_edge(j[0], j[1])
		
		lst.append(TempGraph)
		#print(i)
	#print(lst)
	return lst

#main function
if __name__ == '__main__':
	#dataset = sys.argv[1] #input the original dataset/graph name
	outputfile = 'mindelss_output' #input the output file name
	
	G = []
	paths = []
	levels = [] 
	sources = []
	inedges = []
 
	print("Generating Simulations")
	
	#Change num samples here
	#numSamples = 20
	
	#Change probability of transmission here
	#Used to determine the probability of a given edge being sampled
	p = 0.20

	#Change allowable infection rate here
	alpha = 0.3



	#So this graph is only 75 nodes, 1138 edges.
	df = pd.read_csv('hospital_contacts', sep='\t', header=None)
	df.columns = ['time', 'e1', 'e2', 'lab_1', 'lab_2']
	H = nx.from_pandas_edgelist(df, 'e1', 'e2')

	#This graph is 2500 nodes, 9388 edges, but is randomly constructed
	#H = nx.read_adjlist('random_adj_list')
	
	g = sampling(5, H, p)

	solutions = []
	SimulationSources = []

	#formatting and generating the inputs for the Delay Detection function
	for simulation in g:
		nodess = RandomSolution(simulation, 2)
		source = RandomInfectionSources(simulation, 1)
		solutions.append(nodess)
		sample = (simulation, source)
		SimulationSources.append(sample)

	
	print("solutions: ", solutions, "\n")
	print("Simulation and Sources: " ,SimulationSources, "\n")
	ListOfExpectedDelays = []

	#Determining the delay 
	for solution in solutions:
		for SimandSource in SimulationSources:
			x = DetectDelay(SimandSource[0], solution, SimandSource[1])
			ListOfExpectedDelays.append(x)

	#print(ListOfExpectedDelays)
	expected_delay = np.mean(ListOfExpectedDelays)/(len(solutions))
	print("Delay Calculated")
	print("Expected delay is: ", expected_delay)
