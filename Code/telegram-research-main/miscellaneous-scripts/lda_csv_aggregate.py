import pandas as pd

# Load the data
data = pd.read_csv('topic_modeling_results_lda_gg.csv')

# Convert the 'score' column to a numerical data type
data['score'] = pd.to_numeric(data['score'], errors='coerce')

# Split the 'cluster' column to extract the filename (subdirectory)
data['sub_directory'] = data['cluster'].apply(lambda x: x.split('/')[0])

# Group by 'sub_directory' and compute the average 'score'
average_scores = data.groupby('sub_directory')['score'].mean().reset_index()

# Output the result to a new CSV file
average_scores.to_csv('average_scores_lda_gg.csv', index=False)
