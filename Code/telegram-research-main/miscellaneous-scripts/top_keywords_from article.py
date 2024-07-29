import pandas as pd

# Load your data into a pandas DataFrame. Here I'm assuming your data is in a CSV file.
df = pd.read_csv('keywords.csv')

# Group the data by 'article' and 'cluster_id', and then get the top 5 keywords by 'coherence_score' for each group
result = df.groupby(['article', 'cluster_id']).apply(lambda x: x.nlargest(5, 'coherence_score')['top_words'])

# Reset the index to make the data look nice
result = result.reset_index(drop=True)

# Convert the resulting Series into a DataFrame and assign column names
result_df = pd.DataFrame(result, columns=['article', 'cluster_id', 'top_keywords'])

# Write the result to a new CSV file
result_df.to_csv('top_keywords.csv', index=False)
