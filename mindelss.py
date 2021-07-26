import networkx as nx
import sys
from gurobipy import *
import math
import numpy as np
import time

from collections import defaultdict






def LinearProgram(graph, k, src):
	#graph is a list of sampled subgraphs, generated before being fed in here.
	#src is a list of infection sources. They are fixed for every sample, but there can be multiple.
	#k is a fixed variable

	x = {} #defined as in the paper
	y = {} #idefined as in the paper
	M = len(graph) #number of sample graphs in list graph
	nodes = graph[0].number_of_nodes()

	m = Model('MinDelSS')


	
	#Defines the V_id sets in a dictionary with first key i second key is d, which will return a list of nodes at that level.
	#The way this is constructed, 
	v_set = {}
	for i in range(M):
		v_set[i] = defaultdict([])
		graph[i].add_node('meta')
		for j in src:
			graph[i].add_edge('meta', j)
		#This will return 1 for source nodes and the distance to the rest of them
		paths = nx.shortest_path_length(graph[i], source='meta')
		for u in graph[i]:
			try:
				#paths[u] returns distance d.
				v_set[i][paths[u]].append(u)
				#returns n+1 as the distance if there is an error.
			except nx.exception.NetworkXNoPath:
				v_set[i][nodes + 1].append(u)


	for i in range(M):
		for d in v_set[i].keys():
			for u in v_set[i][d]:
				if u not in x:
					#adding all of the nodes to the x dictionary - only need to add each node once despite all of the samples.
					x[str(u)] = m.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub = 1.0, name = 'x['+str(u)']')
					#x[str(i)+','+str(d)+','+str(u)] = m.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub = 1.0, name = 'x['+str(i)+','+str(d)+','+str(u)']')
			y[str(i)+','+str(d)] = m.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub = 1.0, name = 'y['+str(i)+','+str(d)+']')
	m.update()

	for i in range(M):
		for d in v_set[i].keys():
			m.addConstr(quicksum(x[str(u)] for u in v_set[i][d]) >= y[str(i)+','+str(d)], name='C1_sample='+str(i)+'_dist='+str(d))
	m.update()

	for i in range(M):
		for d in v_set[i].keys():
			m.addConstr(quicksum(x[str(u)] for u in v_set[i][d]) <= k, name='C2_sample='+str(i)+'_dist='+str(d))
	m.update()


	for i in range(M):
		m.addConstr(quicksum(y[str(i)+','+str(d)] for d in v_set[i].keys()) == 1, name='C3_sample='+str(i))

	m.update()




	#Objective Function
	#First fraction is 1/N, need to somehow sum over all d values as well, think that is done in the comprehension right now but maybe need more clarity there
	m.setObjective(1/range(M) * quicksum((y[str(i)+','+str(d)] * d) for i in range(M) for d in v_set[i].keys()), GRB.MINIMIZE)
	m.update()
	m.optimize()

	return m, m.objVal


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
		print(i)
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
	p = 0.02

	#Change allowable infection rate here
	alpha = 0.3

	#for i in range(0, 1):
		#print("Simulation ", i+1)
		#graph, source = Simulation.generateDendograms(H, p, exps)
		#G.append(graph)
		#Took out path from the return on generateDendogram, so commenting out below
		#Not sure where that return came from.
		#paths.append(path)
		#sources.append(source)
	#for i in range(2):
	#H = nx.read_adjlist('adj_list.txt')
	#	G.append(H)

	#H = nx.read_edgelist('lastfm_asia_edges.csv', delimiter=',')
	#H = nx.gaussian_random_partition_graph(5000, 100, 100, .7, .1)
	H = nx.read_adjlist('random_adj_list')
	
	arr = []	

	for i in range(5, 500, 5):
		numSamples = i
		#arr = []
		for j in range(0,1):
			start_time = time.time()
			G = sampling(numSamples, H, p)

			m, x = LinearProgram(G, 'VALUE_FOR_K', 'LIST_OF_SOURCE_NODES')
			tot_time = time.time() - start_time			
			
			arr.append(str(i)+','+str(tot_time)+'\n')

		textfile = open('p_scaling_tot.csv', 'w')
		textfile.write('Number of Samples,Execution Time\n')
		for k in arr:
			textfile.write(str(k))
		textfile.close()
	