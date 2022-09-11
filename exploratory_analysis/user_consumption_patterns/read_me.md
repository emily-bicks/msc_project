To run the experiment "Understanding Learner Consumption Patterns":
1. Specify the number of test users to include as n_users in user_consumption_analysis.py (20 and 50 users were used in the paper)
2. Run user_consumption_analysis.py to generate the data. This will output one file to results/N_USERS_users.json
3. Point consumption_hypothesis_test.ipynb to the file generated in the previous step and run