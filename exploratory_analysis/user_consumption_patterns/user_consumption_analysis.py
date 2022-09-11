import pandas as pd
import networkx as nx
import csv
from collections import defaultdict
import heapq
import numpy as np
import sys
sys.path.insert(1, '../../helper_functions')
import network_utils
from multiprocessing import Pool
from itertools import repeat
import json

def get_community_count(user,userVidLookup,communities):
    user_comms = []
    user_vids = userVidLookup[user]
    for vid in user_vids:
        for key,commun in communities.items():
            if vid in commun:
                user_comms.append(key)
                break
            else:
                continue

    return (user,len(user_vids),len(list(set(user_comms))))

if __name__ == '__main__':

	sample_size = 20
	n_users = 20
	training_data = network_utils.load_peek_train()
	network = network_utils.create_network(training_data)

	test_data = network_utils.load_peek_test()
   
	most_active_users = network_utils.most_active_users(test_data,n_active=20)

	alg_types = ['louvain','label_prop']

	userVidLookup, vidUserLookup = network_utils.create_lookups(test_data)

	final_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

	for alg_type in alg_types:
		communities = network_utils.create_communities(network,alg_type=alg_type)

		communities_with_labels = {i:commun for i,commun in enumerate(communities)}

		community_sizes = [len(commun) for commun in communities]

		with Pool(5) as p:
			results = p.starmap(get_community_count,zip(most_active_users,repeat(userVidLookup),repeat(communities_with_labels)))

		for result in results:
			final_data[alg_type][result[0]]["videos_consumed"].append(result[1])
			final_data[alg_type][result[0]]["actual"].append(result[2])
		print("Completed Run {} Actual".format(alg_type))

		counter = 0

		while counter < sample_size:
			random_communities = network_utils.generate_random_baseline_communities(community_sizes,network)

			random_communities_with_labels = {i:commun for i,commun in enumerate(random_communities)}

			with Pool(5) as p:
				random_results = p.starmap(get_community_count,zip(most_active_users,repeat(userVidLookup),repeat(random_communities_with_labels)))

			for result in random_results:
				final_data[alg_type][result[0]]["random"].append(result[2])

			counter+=1

			print("Completed Random Run {} for {} Algorithm".format(counter,alg_type))

	fn = 'results/{}_users.json'.format(n_users)
	with open(fn,'w') as f:
		json.dump(final_data, f)








		