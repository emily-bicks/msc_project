The scripts in this folder are all part of the recommendations experiments:

- baseline_recs.py: the script to generate both accuracy and non-accuracy metrics of the baseline recommendations. The parameter top_20 at the top of the main routine controls whether to calculate the metrics from the top 20 recommendations, or use the total_set of recommendations.

- community_recs_random_ranking.py: the script to generate both accuracy and non-accuracy metrics of the within community and outside community recommendations with both clustering algorithms using random ranking. The parameter top_20 at the top of the main routine controls whether to calculate the metrics from the top 20 recommendations, or use the total_set of recommendations. The parameter n_hops at the top of the main routine controls whether to define a neighbor as 1 hop or 2 hops away from the most recently consumed video.

- community_recs_user_ranking.py: the script to generate both accuracy and non-accuracy metrics of the within community and outside community recommendations with both clustering algorithms using user ranking. The parameter top_20 at the top of the main routine controls whether to calculate the metrics from the top 20 recommendations, or use the total_set of recommendations. The parameter n_hops at the top of the main routine controls whether to define a neighbor as 1 hop or 2 hops away from the most recently consumed video.

- community_recs_content_ranking.py: the script to generate both accuracy and non-accuracy metrics of the within community and outside community recommendations with both clustering algorithms using content ranking. The parameter top_20 at the top of the main routine controls whether to calculate the metrics from the top 20 recommendations, or use the total_set of recommendations. The parameter n_hops at the top of the main routine controls whether to define a neighbor as 1 hop or 2 hops away from the most recently consumed video.

All of the above scripts create subfolders in the results directory to store the output based on the specified parameters. The results present the accuracy and non-accuracy metrics for each test user.

- Analysis.ipynb: must be pointed at a json file in the results directory. Takes the results of each experiment and presents the aggregate accuracy and non-accuracy metrics. 


