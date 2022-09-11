To run the semantic similarity of communities experiment:
1. Specify the desired similarity metric in similarity_analysis.py by specifying the variable "SIM_METRIC", this must be one of the available similarity metrics provided by the WAT API
2. Run similarity_analysis.py - this will output two results files to the folder results/similarity_data/
	- hypothesis_test_data_SIM_METRIC_louvain.json
	- hypothesis_test_data_SIM_METRIC_label_prop.json
3. Point similarity_hypothesis_test.ipynb to one of the files to generate the p-values
4. Repeat for all similarity metrics provided by the WAT API: https://sobigdata.d4science.org/web/tagme/wat-api