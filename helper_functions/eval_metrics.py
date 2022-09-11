
import math
import networkx as nx


def content_diversity(recommended_ids,network):
	'''
	input: list of ids of the items recommended to the user; lookup of video ids to their associated kcs 
	output: metric of content diversity defined in (Kaminskas and Bridge, 2016) - the average pairwise distance between the recommendations
	'''

	kcs = nx.get_node_attributes(network, "concepts")
	#	 ONLY USE IDS THAT HAVE KCS ASSOCIATED
	kcIds = []
	for recid in recommended_ids:
		try:
			kcs[recid]
			kcIds.append(recid)
		except KeyError:
			continue

	recommended_kcs = [kcs[recommended_id] for recommended_id in kcIds]
	#recommended_kcs = [item for sublist in recommended_kcs_long for item in sublist]
	if len(recommended_kcs) > 1:
		denom = (math.factorial(len(recommended_kcs)))/(2*(math.factorial(len(recommended_kcs)-2)))
		temp = 0
		while len(recommended_kcs)>1:
			i = set(recommended_kcs[0])
			other_items = recommended_kcs[1:]
			for j in other_items:
				dist = 1-((len(list(i.intersection(set(j)))))/(len(list(set(recommended_kcs[0] + j)))))
				temp += dist
			recommended_kcs = other_items

		return temp/denom
	else:
		return 0


def content_serendipity(recommended_ids,user_profile_ids,network):
	'''
	inputs: list of ids of recommended items, and ids of what the user has already consumed
	output: metric of content based serendipity defined in  (Kaminskas and Bridge, 2016) - the average of the minimum pairwise distance between each recommended item and the items in the users profile
	'''
	
	kcs = nx.get_node_attributes(network, "concepts")
	#	 ONLY USE IDS THAT HAVE KCS ASSOCIATED
	kcIds_recommended = []
	for recid in recommended_ids:
		try:
			kcs[recid]
			kcIds_recommended.append(recid)
		except KeyError:
			continue

	
	kcIds_profile= []
	for watched_id in user_profile_ids:
		try:
			kcs[watched_id]
			kcIds_profile.append(watched_id)
		except KeyError:
			continue
	
	

	recommended_kcs = [kcs[recommended_id] for recommended_id in kcIds_recommended]
	#recommended_kcs = list(set([item for sublist in recommended_kcs_long for item in sublist]))

	user_profile_kcs =  [kcs[user_profile_id] for user_profile_id in kcIds_profile]
	#user_profile_kcs = list(set([item for sublist in user_profile_kcs_long for item in sublist]))

	num = 0
	if len(recommended_kcs) >0:
		for i in recommended_kcs:
			i = set(i)
			distances = []
			for j in user_profile_kcs:
				j = set(j)
				dist = 1-((len(list(i.intersection(j))))/(len(list(set(list(i) + list(j))))))
				distances.append(dist)
			num += min(distances)

		return num/len(recommended_ids)
	else:
		return 0



