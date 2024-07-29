import pandas as pd

# Load the data
data = pd.read_csv('topic_modeling_results_nltk_gg.csv')

# Convert the 'score' column to a numerical data type
data['coherence_score'] = pd.to_numeric(data['coherence_score'], errors='coerce')

# Split the 'cluster' column to extract the filename (subdirectory)
data['sub_directory'] = data['filename'].apply(lambda x: x.split('/')[2])

# Group by 'sub_directory' and compute the average 'score'
average_scores = data.groupby('sub_directory')['coherence_score'].mean().reset_index()

# Output the result to a new CSV file
average_scores.to_csv('average_scores_nltk_gg.csv', index=False)
