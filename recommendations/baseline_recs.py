import pandas as pd
import networkx as nx
import csv
from collections import defaultdict
import heapq
import numpy as np
import sys
sys.path.insert(1, '../helper_functions')
import network_utils
import eval_metrics
from multiprocessing import Pool
from itertools import repeat
import json
import statistics as s
import random
import os

def recommend(user,test_data,network, top_20=False):


	test_user_vids = test_data.loc[(test_data['userID']==user) & (test_data['label']==1)].sort_values(by='timestamp',ascending=False)
	test_user_vids['fragmentId'] = None
	for i, row in test_user_vids.iterrows():
		test_user_vids.at[i, 'fragmentId'] = int(str(int(row['vidLectureID'])) + str(int(row['vidID'])) + str(int(row['partID'])))

	user_profile = list(test_user_vids['fragmentId'])
	n_consumed = len(user_profile)

	diversities = []

	serendipities = []

	predictions_counter = 0

	results = defaultdict(dict)

	missing_node_counter = 0

	ranks = []

	rec_totals = []

	for i in range(0,n_consumed-1):
		next_consumed = user_profile[i]
		just_consumed = user_profile[i+1]

		recs = []

		if network.has_node(just_consumed):

			consumed_neighbors = [item[1] for item in nx.bfs_edges(network,source=just_consumed,depth_limit=2)]
			for neighbor in consumed_neighbors:
				recs.append(neighbor)


			if top_20:
				recs = recs[:20]
			diversities.append(eval_metrics.content_diversity(recs,network))

			serendipities.append(eval_metrics.content_serendipity(recs,user_profile,network))

			rec_totals.append(len(recs))

			
			random.shuffle(recs)
			if next_consumed in recs:
				predictions_counter +=1
				ranks.append(1/(1+recs.index(next_consumed)))

		else:
			missing_node_counter+=1

#		else:
#			continue

	diversity_mean = s.mean(diversities)

	serendipity_mean = s.mean(serendipities)

	results["diversity"] = diversity_mean

	results["serendipity"] = serendipity_mean

	results["accuracy"] = predictions_counter/n_consumed

	results["average_total_recs"] = s.mean(rec_totals)

	if len(ranks)>0:
		results["average_inverse_rank"] = s.mean(ranks)
	else:
		results["average_inverse_rank"] = 0

	results["missing_nodes"] = missing_node_counter

	return user,results

if __name__ == '__main__':

	n_active = 20
	top_20 = False

	training_data = network_utils.load_peek_train()
	network = network_utils.create_network(training_data, use_kcs=True)

	test_data = network_utils.load_peek_test()
   
	most_active_users = network_utils.most_active_users(test_data,n_active=n_active)


	final_data = defaultdict()

	with Pool(5) as p:
		results = p.starmap(recommend,zip(most_active_users,repeat(test_data),repeat(network),repeat(top_20)))

	for user, user_data in results:
		final_data[user] = user_data

	if top_20:
		fn = 'results/baseline/top_20.json'.
	else:
		fn = 'results/baseline/total_set.json'

	os.makedirs(os.path.dirname(fn), exist_ok=True)

	with open(fn,'w') as f:
		json.dump(final_data, f)








		