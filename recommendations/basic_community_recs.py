import pandas as pd
import networkx as nx
import csv
from collections import defaultdict
import heapq
import numpy as np
import sys
sys.path.insert(1, '../Helper Functions')
import network_utils
import eval_metrics
from multiprocessing import Pool
from itertools import repeat
import json
import statistics as s
import random

def recommend(user,test_data,communities,network):

	#Find all videos consumed by test user, sort by date, add fragment id to match nodes in network

	test_user_vids = test_data.loc[(test_data['userID']==user) & (test_data['label']==1)].sort_values(by='timestamp',ascending=False)
	test_user_vids['fragmentId'] = None
	for i, row in test_user_vids.iterrows():
		test_user_vids.at[i, 'fragmentId'] = int(str(int(row['vidLectureID'])) + str(int(row['vidID'])) + str(int(row['partID'])))

	user_profile = list(test_user_vids['fragmentId'])
	n_consumed = len(user_profile)

	in_com_diversities = []
	out_com_diversities = []

	in_com_serendipities = []
	out_com_serendipities = []

	in_com_predictions_counter = 0
	out_com_predictions_counter = 0

	results = defaultdict(dict)

	missing_node_counter = 0

	in_com_ranks = []
	out_com_ranks = []

	in_com_rec_totals = []
	out_com_rec_totals = []

	for i in range(0,n_consumed-1):
		next_consumed = user_profile[i]
		just_consumed = user_profile[i+1]

		in_com_recs = []
		out_com_recs = []

		if network.has_node(just_consumed):
			for comid, comm in enumerate(communities):
				if just_consumed in comm:
					comid = comid
					comm = comm
					break
				else:
					continue
			consumed_neighbors = [item[1] for item in nx.bfs_edges(network,source=just_consumed,depth_limit=2)]
			for neighbor in consumed_neighbors:
				if neighbor in comm:
					in_com_recs.append(neighbor)
				else:
					out_com_recs.append(neighbor)
		
			random.shuffle(in_com_recs)
			random.shuffle(out_com_recs)

			in_com_recs = in_com_recs[0:20]
			out_com_recs = out_com_recs[0:20]

			if next_consumed in in_com_recs:
				in_com_predictions_counter +=1
				in_com_ranks.append(1/(1+in_com_recs.index(next_consumed)))
			elif next_consumed in out_com_recs:
				out_com_predictions_counter +=1
				out_com_ranks.append(1/(1+out_com_recs.index(next_consumed)))
			#else:
			#	continue


			#print(in_com_recs)

			in_com_diversities.append(eval_metrics.content_diversity(in_com_recs,network))
			out_com_diversities.append(eval_metrics.content_diversity(out_com_recs,network))

			in_com_serendipities.append(eval_metrics.content_serendipity(in_com_recs,user_profile,network))
			out_com_serendipities.append(eval_metrics.content_serendipity(out_com_recs,user_profile,network))

			out_com_rec_totals.append(len(out_com_recs))
			in_com_rec_totals.append(len(in_com_recs))

		else:
			missing_node_counter+=1


	if len(out_com_diversities)<1:
		print(in_com_recs)
		print(out_com_recs)
		print(user)
		print(n_consumed)
	outcom_diversity_mean = s.mean(out_com_diversities)
	in_com_diversity_mean = s.mean(in_com_diversities)

	outcom_serendipity_mean = s.mean(out_com_serendipities)
	in_com_serendipity_mean = s.mean(in_com_serendipities)

	results["in_community"]["diversity"] = in_com_diversity_mean

	results["in_community"]["serendipity"] = in_com_serendipity_mean

	results["in_community"]["accuracy"] = in_com_predictions_counter/n_consumed

	results["in_community"]["average_total_recs"] = s.mean(in_com_rec_totals)

	if len(in_com_ranks)>0:		

		results["in_community"]["average_inverse_rank"] = s.mean(in_com_ranks)

	else:

		results["in_community"]["average_inverse_rank"] = 0

	results["out_community"]["diversity"] = outcom_diversity_mean

	results["out_community"]["serendipity"] = outcom_serendipity_mean

	results["out_community"]["accuracy"] = out_com_predictions_counter/n_consumed

	results["out_community"]["average_total_recs"] = s.mean(out_com_rec_totals)

	if len(out_com_ranks)>0:

		results["out_community"]["average_inverse_rank"] = s.mean(out_com_ranks)

	else:
		results["out_community"]["average_inverse_rank"] = 0

	results["missing_nodes"] = missing_node_counter

	return user,results

if __name__ == '__main__':

	n_active = 20

	training_data = network_utils.load_peek_train()
	network = network_utils.create_network(training_data, use_kcs=True)

	test_data = network_utils.load_peek_test()
   
	most_active_users = network_utils.most_active_users(test_data,n_active=n_active)

	alg_types = ['louvain','label_prop']


	final_data = defaultdict()

	for alg_type in alg_types:
		communities = network_utils.create_communities(network,alg_type=alg_type, minNodeSize=0)

		final_data = defaultdict(dict)
		with Pool(5) as p:
			results = p.starmap(recommend,zip(most_active_users,repeat(test_data),repeat(communities),repeat(network)))

		for user, user_data in results:
			final_data[user] = user_data

		fn = 'results/{}_{}.json'.format(alg_type,n_active)
		with open(fn,'w') as f:
			json.dump(final_data, f)








		