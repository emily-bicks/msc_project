import pandas as pd
import networkx as nx
import csv
from collections import defaultdict
from heapq import nlargest
#from cdlib import algorithms
import random
from itertools import islice
import json
from scipy.sparse import dok_array, lil_matrix

def load_peek_train():
	'''
	load in and return a dataframe of the peek training data 
	'''

	return pd.read_csv('..//..//data//train.csv', names = ['vidLectureID','vidID','partID','timestamp','userID',
										 'KCID','topicCoverage','KCID2','topicCoverage2','KCID3','topicCoverage3',
										 'KCID4','topicCoverage4','KCID5','topicCoverage5','label'])

def load_peek_test():
	'''
	load in and return a dataframe of the peek test data 

	'''

	return pd.read_csv('..//..//data//test.csv', names = ['vidLectureID','vidID','partID','timestamp','userID',
										 'KCID','topicCoverage','KCID2','topicCoverage2','KCID3','topicCoverage3',
										 'KCID4','topicCoverage4','KCID5','topicCoverage5','label'])

def create_lookups(watched_fragments):
	'''
	create and return two dictionaries:
	1. userVidLookup: key: user, value: all videos consumed by user
	2. vidUserLookup: key: video, value: all users who have consumed the video
	'''

	userVidLookup = defaultdict(list)
	vidUserLookup = defaultdict(list)
	for index,row in watched_fragments.iterrows():
		vidID = int(str(int(row['vidLectureID'])) + str(int(row['vidID'])) + str(int(row['partID'])))
		userVidLookup[int(row['userID'])].append(vidID)
		vidUserLookup[vidID].append(int(row['userID']))

	return userVidLookup, vidUserLookup

'''
def create_random_lookup(userVidLookup,users,network):

	unique_fragments = list(network.nodes)
	randomLookup = defaultdict(dict)

	for user in users:
		n_fragments = len(userVidLookup[user])

		userVids = random.sample(unique_fragments,n_fragments)

		randomLookup[user] = userVids

	return randomLookup


def clean_kcs():

	idToKCLookup = defaultdict(tuple)
	with open("..//data//id_to_wiki_url_mapping.csv", 'r') as f:
		csvreader = csv.reader(f)
		next(csvreader)
		for line in csvreader:
			last_url_snip = line[1].split("/")[-1]
			last_url_snip_clean = ''.join([i if ord(i) < 128 else ' ' for i in last_url_snip]) #replace non ascii characters with a space			
			try:
				title = wikipedia.search(last_url_snip_clean)[0]
				page = wikipedia.page(title=title)
				pageid = page.pageid
				idToKCLookup[line[0]] = (title,pageid)
			except:
				idToKCLookup[line[0]] = (title)
			
	return idToKCLookup
'''

def user_vectors(vidUserLookup,vid,userPositions):
	#vec = dok_array((2, len(list(userPositions.keys()))), dtype=int)
	vec = lil_matrix((1, len(list(userPositions.keys()))), dtype=int)

	for user in vidUserLookup[vid]:
		vec[0,userPositions[user]] = 1

	return vec.tocsr()


def create_network(data, use_kcs=False, useUserVecs = False):
	'''
	Take data and create a network where nodes are video fragments and edges are all users who have consumed both lectures. If use_kcs=True, add node attribute with a list of all kcs associated with the video fragment.
	'''

	

	watched_fragments = data[data['label']==1]
	unique_fragments = watched_fragments.drop_duplicates(subset=['vidLectureID','vidID','partID'])
	print("There are {} unique, watched video fragments".format(len(unique_fragments)))
	

	userVidLookup, vidUserLookup = create_lookups(watched_fragments)

	network = nx.Graph()


	if (use_kcs and useUserVecs):
		with open('..//data//wikiLookup.json','r') as f:
			idToKCLookup = json.load(f)

		userPositions = list(enumerate(userVidLookup.keys()))
		userPositionsDict = {key:val for val,key in userPositions}

		for i,row in unique_fragments.iterrows():
			nodeId = int(str(int(row['vidLectureID'])) + str(int(row['vidID'])) + str(int(row['partID'])))
			concepts = []
			conceptIds = []
			try:
				concepts.append(idToKCLookup[str(int(row['KCID']))][0])
				concepts.append(idToKCLookup[str(int(row['KCID2']))][0])
				concepts.append(idToKCLookup[str(int(row['KCID3']))][0])
				concepts.append(idToKCLookup[str(int(row['KCID4']))][0])
				concepts.append(idToKCLookup[str(int(row['KCID5']))][0])
				conceptIds.append(idToKCLookup[str(int(row['KCID']))][1])
				conceptIds.append(idToKCLookup[str(int(row['KCID2']))][1])
				conceptIds.append(idToKCLookup[str(int(row['KCID3']))][1])
				conceptIds.append(idToKCLookup[str(int(row['KCID4']))][1])
				conceptIds.append(idToKCLookup[str(int(row['KCID5']))][1])
			except:
				continue
			concepts = list(set(concepts))
			conceptIds = list(set(conceptIds))
			userVector = user_vectors(vidUserLookup,nodeId,userPositionsDict)
			network.add_node(nodeId,id=nodeId,vidLectureID = int(row['vidLectureID']),vidID = int(row['vidID']),partID = int(row['partID']),concepts=concepts,conceptIds = conceptIds, userVector = userVector)
	
	elif use_kcs:
		#idToKCLookup = clean_kcs()
		with open('..//data//wikiLookup.json','r') as f:
			idToKCLookup = json.load(f)

		for i,row in unique_fragments.iterrows():
			nodeId = int(str(int(row['vidLectureID'])) + str(int(row['vidID'])) + str(int(row['partID'])))
			concepts = []
			conceptIds = []
			try:
				concepts.append(idToKCLookup[str(int(row['KCID']))][0])
				concepts.append(idToKCLookup[str(int(row['KCID2']))][0])
				concepts.append(idToKCLookup[str(int(row['KCID3']))][0])
				concepts.append(idToKCLookup[str(int(row['KCID4']))][0])
				concepts.append(idToKCLookup[str(int(row['KCID5']))][0])
				conceptIds.append(idToKCLookup[str(int(row['KCID']))][1])
				conceptIds.append(idToKCLookup[str(int(row['KCID2']))][1])
				conceptIds.append(idToKCLookup[str(int(row['KCID3']))][1])
				conceptIds.append(idToKCLookup[str(int(row['KCID4']))][1])
				conceptIds.append(idToKCLookup[str(int(row['KCID5']))][1])
			except:
				continue
			concepts = list(set(concepts))
			conceptIds = list(set(conceptIds))
			network.add_node(nodeId,id=nodeId,vidLectureID = int(row['vidLectureID']),vidID = int(row['vidID']),partID = int(row['partID']),concepts=concepts,conceptIds = conceptIds)

	elif useUserVecs:

		userPositions = list(enumerate(userVidLookup.keys()))
		userPositionsDict = {key:val for val,key in userPositions}
		for i,row in watched_fragments.iterrows():

			nodeId = int(str(int(row['vidLectureID'])) + str(int(row['vidID'])) + str(int(row['partID'])))
			userVector = user_vectors(vidUserLookup,nodeId,userPositionsDict)

			network.add_node(nodeId,id=nodeId,vidLectureID = int(row['vidLectureID']),vidID = int(row['vidID']),partID = int(row['partID']),userVector = userVector)


	else:
		for i,row in watched_fragments.iterrows():

			nodeId = int(str(int(row['vidLectureID'])) + str(int(row['vidID'])) + str(int(row['partID'])))
			network.add_node(nodeId,id=nodeId,vidLectureID = int(row['vidLectureID']),vidID = int(row['vidID']),partID = int(row['partID']))


	for user,vids in userVidLookup.items():
		if len(vids)>1:
			for vid in vids:
				sublist = [v for v in vids if v != vid]
				for i in range(0,len(sublist)):
					if network.has_edge(vid,sublist[i]):
						network[vid][sublist[i]]["weight"]+=1
					else:
						network.add_edge(vid,sublist[i],weight=1)
				vids.remove(vid)

	return network

def generate_random_baseline_communities(community_sizes,network,seed=None):
	'''
	Randomly sample specified number of nodes from the network to
	Input: list of community sizes for sample (based on sizes of communities from a given algorithm)
	Output: A list of "random communities"  
	'''
	if seed is not None:
		random.seed(seed)  #set seed to get the same random shuffle each time
		network_nodes = iter(random.sample(list(network.nodes), len(network.nodes)))
	else:
		network_nodes = iter(random.sample(list(network.nodes), len(network.nodes)))

	return [list(islice(network_nodes, elem))for elem in community_sizes]

def create_communities(network, alg_type = 'label_prop',minNodeSize = 10):
	'''
	Generate a list of communities. Each community contains a list of all nodes in that community.
	MinNodeSize corresponds to the number of videos that need to be present in a community to consider it.
	'''
	if alg_type == 'label_prop':
		communities =  list(nx.algorithms.community.asyn_lpa_communities(network,weight='weight',seed=42))
	elif alg_type == 'louvain':
		communities = nx.algorithms.community.louvain_communities(network,weight='weight',seed=42)
	#elif alg_type == 'walktrap':
	#	communities = algorithms.walktrap(network).communities
	else:
		print("INVALID ALGORITHM TYPE")

	big_communities=[]
	for item in communities:
		if len(item)>minNodeSize:
			big_communities.append(item)
	return big_communities

def most_active_users(data,n_active=20,all_users=False):
	'''
	Given a dataset (test or train), generate a list of the n_active most active users in the dataset, where active is defined as consuming the most unique video lecture fragments.
	'''
	if all_users:
		return list(set(data['userID']))
	else:
		temp = data['userID'].value_counts()
		return list(temp.index[0:n_active])


def getKCs(community,network,N=5):
	'''
	Get the most frequent N KCs in a given community
	'''
	subg = nx.subgraph(network,community)
	concept_counter = defaultdict(int)
	concepts = nx.get_node_attributes(subg, "concepts")
	for node,item in concepts.items():
		for concept in item:
			concept_counter[concept]+=1
	return nlargest(N, concept_counter, key = concept_counter.get)

def getKCIds(community,network,N=5):
	'''
	Get the most frequent N KC ids in a given community
	'''
	subg = nx.subgraph(network,community)
	concept_counter = defaultdict(int)
	conceptIds = nx.get_node_attributes(subg, "conceptIds")
	for node,item in conceptIds.items():
		for concept in item:
			concept_counter[concept]+=1
	return nlargest(N, concept_counter, key = concept_counter.get)



