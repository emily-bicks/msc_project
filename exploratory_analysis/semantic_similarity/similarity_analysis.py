import pandas as pd
import networkx as nx
import csv
from collections import defaultdict
import heapq
import numpy as np
import sys
sys.path.insert(1, '../Helper Functions')
import network_utils
import matplotlib.pyplot as plt
import requests
from itertools import combinations,repeat
import json
from multiprocessing import Pool

import statistics

def get_community_score(community,network, sim_metric):
	kcs = network_utils.getKCIds(community,network,N=5) # get the top 10 most frequently occuring KCs associated with the community
	within_commun_total = 0
	kcCombos = list(combinations(kcs,2))
		
	url = 'https://wat.d4science.org/wat/relatedness/graph'
	token = '0e8376a1-75a1-4d14-9df6-5bb8fdaab675-843339462'

	for pair in kcCombos:
		payload = [("gcube-token",token),("relatedness",sim_metric),("ids",pair[0]),("ids",pair[1])] 
		response = requests.get(url, params=payload)
		score = json.loads(response.text)['pairs'][0]['relatedness']
		within_commun_total += score
	return (len(community),within_commun_total/len(kcCombos))



if __name__ == '__main__':

	
	sample_size = 10 #SPECIFY HOW MANY RANDOM RUNS 

	# Create network using peek training data
	training_data = network_utils.load_peek_train()
	network = network_utils.create_network(training_data, use_kcs=True)

	sim_metric = 'pmi'

	# Generate communities using specified algorithm
	alg_types = ['louvain','label_prop']
	for alg_type in alg_types:
		communities = network_utils.create_communities(network,alg_type=alg_type,minNodeSize=5) #CONSIDER MIN NODE SIZE CHOICE AND DISCUSS WITH SAHAN/MARIA
		print("{} produced {} communities".format(alg_type,len(communities)))

		url = 'https://wat.d4science.org/wat/relatedness/graph'

		output_dict = defaultdict(list)

		##################################
		# CALCULATE SIMILARITY SCORES FOR COMMUNITIES GENERATED BY SPECIFIED ALGORITHM
		##################################

		with Pool(5) as p:
			scores = p.starmap(get_community_score,zip(communities,repeat(network)))
		
		commun_sizes = [item[0] for item in scores]
		commun_scores = [item[1] for item in scores]

		total_nodes = sum(commun_sizes)
		weights = [size/total_nodes for size in commun_sizes]
		weighted_avg = (sum([weights[i]*commun_scores[i] for i in range(len(commun_scores))])/sum(weights))

		print("Overall Average Score: {}".format(statistics.mean(commun_scores)))
		print("Median Score: {}".format(statistics.median(commun_scores)))
		print("Weighted Average by Community Size: {}".format(weighted_avg))

		plt.figure(0)
		plt.scatter(commun_sizes,commun_scores)
		plt.savefig('results/similarity_plots/{}_{}.png'.format(sim_metric,alg_type))

		plt.clf()

		output_dict['community_average'].append(statistics.mean(commun_scores))
		output_dict['community_weighted_average'].append(weighted_avg)
		

		##################################
		# CALCULATE SIMILARITY SCORES FOR RANDOMLY COMMUNITIES OF THE SAME SIZE DISTRIBUTION AS FROM ALGORITHM
		##################################
		
		counter = 0

		while counter < sample_size:
			random_communities = network_utils.generate_random_baseline_communities(commun_sizes,network)

			with Pool(5) as p:
				random_scores = p.starmap(get_community_score,zip(random_communities,repeat(network), repeat(sim_metric)))

			random_commun_sizes = [item[0] for item in random_scores]
			random_commun_scores = [item[1] for item in random_scores]

			random_weights = [size/total_nodes for size in random_commun_sizes]
			random_weighted_avg = (sum([random_weights[i]*random_commun_scores[i] for i in range(len(random_commun_scores))])/sum(random_weights))

			output_dict["averages"].append(statistics.mean(random_commun_scores))
			output_dict["weighted_averages"].append(random_weighted_avg)

			plt.figure(1)
			plt.scatter(random_commun_sizes,random_commun_scores)
			plt.savefig('results/similarity_plots/{}_{}_{}_baseline.png'.format(sim_metric,alg_type,counter))

			print("FINISHED RUN {}".format(counter))

			counter +=1

		with open('results/similarity_data/hypothesis_test_data_{}_{}.json'.format(sim_metric,alg_type), 'w') as fp:
			json.dump(output_dict, fp)

		plt.clf()



