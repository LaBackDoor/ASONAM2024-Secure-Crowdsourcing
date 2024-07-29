import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA


def lda_topic_modeling(input_folder, output_file, n_topics, n_top_words):
    """
    This function performs LDA-based topic modeling on the clusters in the input directory
    and saves the results into a csv file.

    :param input_folder: str, path to the input directory containing cluster csv files
    :param output_file: str, path to the output csv file to save the results
    :param n_topics: int, number of topics to identify per cluster
    :param n_top_words: int, number of top words to include in each topic
    """

    # Define the function to print the top words for each topic
    def print_top_words(model, feature_names, dtm, n_top_words):
        topics = []
        topic_proportions = model.transform(dtm)
        for topic_idx, topic in enumerate(model.components_):
            message = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            topics.append(message)
        return topics, topic_proportions

    # Initialize an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['cluster', 'topic', 'top_words', 'score'])

    # Loop through each file in the input folder and its subdirectories
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".csv"):
                file_path = os.path.join(root, filename)

                # Extract subdirectory name from the file path
                subdirectory = os.path.relpath(root, input_folder)

                # Load CSV file into a DataFrame
                df = pd.read_csv(file_path)

                # Replace NaN values with an empty string
                df['message'] = df['message'].fillna('')

                # Use CountVectorizer to transform the text data
                vectorizer = CountVectorizer(stop_words='english')
                dtm = vectorizer.fit_transform(df['message'])

                # Apply LDA to the document-term matrix
                lda = LDA(n_components=n_topics, random_state=1).fit(dtm)

                # Get the top words for each topic
                dtm_feature_names = vectorizer.get_feature_names_out()
                topics, topic_proportions = print_top_words(lda, dtm_feature_names, dtm, n_top_words)

                # Write the topics and their proportions to the results DataFrame
                for i, (topic, proportions) in enumerate(zip(topics, topic_proportions.T)):
                    for j, proportion in enumerate(proportions):
                        results_df.loc[len(results_df)] = [subdirectory + '/' + filename.replace('.csv', ''), i + 1,
                                                           topic, proportion]

    # Write the results DataFrame to a CSV file
    results_df.to_csv(output_file, index=False)


# Use the function
lda_topic_modeling("../clustering_keyword_search", "topic_modeling_results_lda_gg.csv", 5, 10)
