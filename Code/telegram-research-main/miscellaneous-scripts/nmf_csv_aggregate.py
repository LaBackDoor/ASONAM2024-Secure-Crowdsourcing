import pandas as pd

# Load the data
data = pd.read_csv('topic_modeling_results_nmf_gg.csv')

# Split the 'cluster' column to extract the filename (subdirectory)
data['sub_directory'] = data['cluster'].apply(lambda x: x.split('/')[2])

# Group by 'sub_directory' and compute the average 'score'
average_scores = data.groupby('sub_directory')['average_score'].mean().reset_index()

# Output the result to a new CSV file
average_scores.to_csv('average_scores_nmf_gg.csv', index=False)
