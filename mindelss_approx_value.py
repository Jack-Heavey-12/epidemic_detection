import networkx as nx
import sys
from gurobipy import *
import math
import numpy as np
import time
import pandas as pd
import pickle


from collections import defaultdict

from matplotlib import pyplot as plt




def rounding(x, n, big_n):
	c = 2 + math.log(big_n, n+1)
	d = c * (math.log(n+1) ** 2)

	limit = np.random.uniform(0,1)

	#just a slightly cleaner way of doing the math, the other way should have still worked, though
	if limit <= np.min([1, x*d]):
		return 1
	return 0




def LinearProgram(graph, k, sources):
	#graph is a list of sampled subgraphs, generated before being fed in here.
	#src is a list of infection sources. They are fixed for every sample, but there can be multiple.
	#UPDATE - This is defined later
	#k is a fixed variable

	x = {} #defined as in the paper
	y = {} #idefined as in the paper
	M = len(graph) #number of sample graphs in list graph
	nodes = graph[0].number_of_nodes()

	src = sources

	#could condense this, but want to make sure that it's the same list so it truly is random
	#could also be faster I suppose at the cost of memory
	#lst_of_nodes = list(graph[0].nodes())

	#makes the same source across all samples
	#val = np.random.randint(0, nodes-1)
	#for i in range(len(graph)):
		#picks a random value out of the number of nodes
		#Uncomment this line for a different source for every sample
		#val = np.random.randint(0, nodes-1)
		#assigns that node as a source
		#src.append(lst_of_nodes[val])


	m = Model('MinDelSS')


	
	#Defines the V_id sets in a dictionary with first key i second key is d, which will return a list of nodes at that level.
	#The way this is constructed, 
	v_set = {}
	for i in range(M):
		empt_lst = lambda:[]
		v_set[i] = defaultdict(empt_lst)
		graph[i].add_node('meta')
		graph[i].add_edge('meta', src[i])
		#This will return 1 for source nodes and the distance to the rest of them
		paths = nx.shortest_path_length(graph[i], source='meta')
		for u in graph[i]:
			try:
				#paths[u] returns distance d.
				v_set[i][paths[u]].append(u)
				#returns n+1 as the distance if there is an error.
			except KeyError:
				v_set[i][nodes + 1].append(u)
		del v_set[i][0]
		graph[i].remove_node('meta')


	for i in range(M):
		for d in v_set[i].keys():
			for u in v_set[i][d]:
				if u not in x:
					#adding all of the nodes to the x dictionary - only need to add each node once despite all of the samples.
					x[str(u)] = m.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub = 1.0, name = 'x['+str(u)+']')
					#x[str(i)+','+str(d)+','+str(u)] = m.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub = 1.0, name = 'x['+str(i)+','+str(d)+','+str(u)']')
			y[str(i)+','+str(d)] = m.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub = 1.0, name = 'y['+str(i)+','+str(d)+']')
	m.update()

	for i in range(M):
		for d in v_set[i].keys():
			m.addConstr(quicksum(x[str(u)] for u in v_set[i][d]) >= y[str(i)+','+str(d)], name='C1_sample='+str(i)+'_dist='+str(d))
	m.update()

	'''for i in range(M):
		for d in v_set[i].keys():
			m.addConstr(quicksum(x[str(u)] for u in v_set[i][d]) <= k, name='C2_sample='+str(i)+'_dist='+str(d))
	m.update()'''

	m.addConstr(quicksum(x[str(u)] for u in list(x.keys())) <= k, name='C2_budget')


	for i in range(M):
		m.addConstr(quicksum(y[str(i)+','+str(d)] for d in v_set[i].keys()) == 1, name='C3_sample='+str(i))

	m.update()

	#Objective Function
	#First fraction is 1/N, need to somehow sum over all d values as well, think that is done in the comprehension right now but maybe need more clarity there
	m.setObjective(1/M * quicksum((y[str(i)+','+str(d)] * d) for i in range(M) for d in v_set[i].keys()), GRB.MINIMIZE)
	m.update()
	m.optimize()



	#x is the dictionary where the values of x are stored, producing the rounding here
	n = len(x)
	x_prime_dict = {}
	for j in x.keys():
		x_prime_dict[j] = rounding(x[j].x, n, num_samples)


	#producing the number of infections here by saving the length of the connected component to the selected source
	infection_list = []
	for i in range(M):
		infection_list.append(len(nx.node_connected_component(graph[i], src[i])))



	#Want to calculate the objective value now for the rounded values
	#first, create a smaller set which has all of the nodes where they are in s_r
	s_r = set()
	for i in list(x_prime_dict.keys()):
		if x_prime_dict[i] == 1:
			s_r.add(i)


	obj_vals = []
	for i in range(M):
		paths = nx.shortest_path_length(graph[i], source=src[i])
		shortest = nodes + 1
		for u in list(s_r):
			try:
				if paths[u] < shortest:
					shortest = paths[u]
			except KeyError:
				continue
		obj_vals.append(shortest + 1)



	#Return stuff here, so want to calculate everything first so that we can return everything with just one function call


	return m, m.objVal, x, x_prime_dict, np.mean(infection_list), np.mean(obj_vals), y, src, v_set


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

	lst_of_nodes = list(graph.nodes())
	sources = []
	nodes = len(graph.nodes())	

	for i in range(num_samples):
		val = np.random.randint(0, nodes-1)
		sources.append(lst_of_nodes[val])

	return lst, sources



def produce_plot(input_name, output_string):
	df = pd.read_csv(input_name)

	df['Percent'] = df['Rounded X Value'] / df['Value of K']

	fig = plt.plot(df['Value of K'], df['Value of K'], color='r', marker='o')

	plt.plot(df['Value of K'], df['Rounded X Value'], color='b', marker='o')
	plt.legend(['Value of Budget K', 'Size of Selected Set S_r'])
	plt.title('Violation of Budget Constraints')

	plt.savefig(output_string + '.png')
	plt.clf()

	fig2 = plt.plot(df['Value of K'], df['Percent'], 'o')
	plt.title('Violation of K as a Percentage of K')
	plt.savefig(output_string +'_percent.png')
	plt.clf()


	fig3 = plt.plot(df['Rounded X Value'], df['new_objVal'], color='m', marker='o')
	plt.title('Size of Sensor Set vs. Rounded Mean Detection Time')
	plt.savefig(output_string + '_objvK.png')
	plt.clf()


#main function
if __name__ == '__main__':
	#dataset = sys.argv[1] #input the original dataset/graph name
	outputfile = 'mindelss_output' #input the output file name
	
	#G = []
	paths = []
	levels = [] 
	sources = []
	inedges = []
 
	print("Generating Simulations")
	
	#Change num samples here
	#numSamples = 20
	
	#Change probability of transmission here
	#Used to determine the probability of a given edge being sampled
	p = 0.15

	#Change allowable infection rate here
	alpha = 0.3



	#So this graph is only 75 nodes, 1138 edges.
	#df = pd.read_csv('hospital_contacts', sep='\t', header=None)
	#df.columns = ['time', 'e1', 'e2', 'lab_1', 'lab_2']
	#H = nx.from_pandas_edgelist(df, 'e1', 'e2')

	#This graph is 2500 nodes, 9388 edges, but is randomly constructed
	#H = nx.read_adjlist('random_adj_list')

	#theoretical physics collaboration network
	#8638 nodes, 24827 edges
	#H_prime = nx.read_adjlist('arxiv_collab_network.txt')
	#H = H_prime.subgraph(max(nx.connected_components(H_prime))).copy()
	#del H_prime
	
	#UVA Hospital Network, 30 weeks from end. 9949 nodes, 399495 edges
	#network = open('personnetwork_post', 'r')
	#lines = network.readlines()
	#lst = []
	#for line in lines:
	#	lst.append(line.strip())
	#network.close()
	#H = nx.parse_edgelist(lst)
	#del lst

	#UVA hospital Network, 30 weeks from beginning (skipping to at least time 10,000). 10789 nodes, 291881 edges
	#network = open('personnetwork_exp', 'r')
	#lines = network.readlines()
	#lst = []
	#for line in lines:
	#	lst.append(line.strip())
	#network.close()
	#H = nx.parse_edgelist(lst)
	#del lst

	#11413 nodes, 25663 edges
	with open('Carilion_1hour.pkl', 'rb') as pkl:
		H = pickle.load(pkl)

	nodes = len(H.nodes()) #small n

	k_arr = []
	obj_val_lst = []
	x_prime_dict_arr = []
	new_obj_val_lst = []
	obj_val_NEW_lst = []

	prob_lst = [.01, .05, .1, .15, .2, .25]

	num_samples = 2500
	G, sources = sampling(num_samples, H, p)

	for i in range(25, 275, 25):
	#for i in prob_lst:
		k = i #VALUE_FOR_K
		#k = np.floor((i/1000) * nodes)
		#p = i
		

		#UPDATE - Deleted LIST_OF_SOURCES as an argument, defined in function randomly.
		model, obj_val, x_dict, x_prime_dict, inf_list, new_obj_val, y_vals, src, v_set = LinearProgram(G, k, sources)

		_, obj_val_NEW, x_dict_NEW, x_prime_dict_NEW, _, _, _, _, _ = LinearProgram(G, sum(x_prime_dict.values()), sources)




		#NEED TO EVALUATE THIS WRITE STATEMENT BELOW HERE
		k_arr.append(k)
		x_prime_dict_arr.append(sum(x_prime_dict.values()))
		obj_val_lst.append(obj_val)
		new_obj_val_lst.append(new_obj_val)
		obj_val_NEW_lst.append(obj_val_NEW)


		textfile = open('approximation_values_output/approximation_carilion.csv', 'w')
		textfile.write('Value of K,Rounded X Value,Original objVal,Rounded objVal,objVal Second Run\n')
		for val in range(len(k_arr)):
			textfile.write(str(k_arr[val])+','+str(x_prime_dict_arr[val])+','+str(obj_val_lst[val])+
				','+str(new_obj_val_lst[val])+','+str(obj_val_NEW_lst[val])+'\n')
		textfile.close()


		'''y_file = open('y_vals.csv', 'w')
		y_file.write('Var_name,Optimized_Value\n')
		for val in list(y_vals.keys()):
			y_file.write(str(val)+','+str(y_vals[val].x)+'\n')
		y_file.close()'''

		'''x_vals = open('approximation_values_output/x_vals_pre_approx_'+str(k)+'_preSUPPLEMENT.csv', 'w')
		x_vals.write('Var_name,Optimized_Value\n')
		for val in list(x_dict.keys()):
			x_vals.write('x['+str(val)+'],'+str(x_dict[val])+'\n')
		x_vals.close()

		x_vals2 = open('approximation_values_output/x_vals_pre_approx_2_'+str(k)+'_preSUPPLEMENT.csv', 'w')
		x_vals2.write('Var_name,Optimized_Value\n')
		for val in list(x_dict_NEW.keys()):
			x_vals2.write('x['+str(val)+'],'+str(x_dict_NEW[val].X)+'\n')
		x_vals2.close()'''

		'''v_file = open('v_file.tsv', 'w')
		v_file.write('Sample\tDistance\tSource\tValues\n')
		for i in range(len(G)):
			for d in list(v_set[i].keys()):
				v_file.write(str(i)+'\t'+str(d)+'\t'+str(src[i])+'\t'+str(v_set[i][d])+'\n')
		v_file.close()'''




	#CAN CHANGE WHAT YOU WANT THE PLOT TO BE CALLED HERE
	#produce_plot('k_values_post_p15.csv', 'post_covid_p15')

	print('Done!')

	