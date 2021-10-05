import networkx as nx
import sys
from gurobipy import *
import math
import numpy as np
import time
import pandas as pd
import random
import pickle

print('beginning program')

def greedy(G, k, sources):
	print('Greedy Start')
	node_lst = list(G[0].nodes())
	node_num = len(node_lst)
	s_r = set()


	#Greedy algorithm is slow - implementation is O(k*n*N*k)
	G_dists = {}
	for budget in range(k):
		min_detection_val = (np.inf, 0)
		for i in node_lst:
			print(i)
			if i in s_r:
				continue
			#s_r.add(i)
			lst = []
			for j in range(len(G)):
				shortest = node_num + 1
				try:
					paths = nx.shortest_path_length(G[j], source=sources[j])
				except Exception:
					G[j].add_node(sources[j])
				try:
					shortest = min(G_dists[j], paths[i])
				except KeyError:
					try:
						shortest = paths[i]
					except KeyError:
						pass
					#G_dists[j] = shortest
				'''for u in s_r:
					paths = nx.shortest_path_length(G[j], source=sources[j])
					try:
						if paths[u]+1 < shortest:
							shortest = paths[u]+1
					except KeyError:
						continue'''
				lst.append(shortest)
			mean = np.mean(lst)
			#s_r.remove(i)
			if mean < min_detection_val[0]:
				min_detection_val = (mean, i)
				for j in range(len(G)):
					G_dists[j] = lst[j]
		s_r.add(min_detection_val[1])


	print('Greedy End')
	return s_r


G = []
sources = []
file = open('baselines_files_5/sources', 'r')
lines = file.readlines()
for line in lines:
	sources.append(line.strip())

for i in range(3000):
	G.append(nx.read_edgelist('baselines_files_5/graphs/sample_graph_'+str(i)))

k = list(range(25,275,25))

print('Files Loaded')

obj_values_greedy = []
for budget in k:

	sr = greedy(G, budget, sources)


	obj_vals_greedy = []
	for i in range(M):
		#paths = nx.shortest_path_length(graph[i], source=src[i])
		shortest = nodes + 1
		for u in sr:
			try:
				if paths[u]+1 < shortest:
					shortest = paths[u]
			except KeyError:
				continue
		obj_vals_greedy.append(shortest)

	obj_values_greedy.append(np.mean(shortest))

output_file = open('baselines_files_5/greedy_output', 'w')
for i in range(len(sources)):
	output_file.write(str(obj_values_greedy[i])+'\n')
