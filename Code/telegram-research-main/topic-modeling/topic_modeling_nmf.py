import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import warnings
from sklearn.exceptions import ConvergenceWarning


def nmf_topic_modeling(input_folder, output_file, n_topics, n_top_words):
    """
    This function performs NMF-based topic modeling on the clusters in the input directory
    and saves the results into a csv file.

    :param input_folder: str, path to the input directory containing cluster csv files
    :param output_file: str, path to the output csv file to save the results
    :param n_topics: int, number of topics to identify per cluster
    :param n_top_words: int, number of top words to include in each topic
    """

    # Define the function to print the top words for each topic
    def print_top_words(model, feature_names, n_top_words):
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            message = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            topics.append(message)
        return topics

    # Initialize an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['cluster', 'topic', 'top_words', 'average_score'])

    # Ignore ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Loop through each file in the input folder and its subdirectories
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".csv"):
                file_path = os.path.join(root, filename)

                # Load CSV file into a DataFrame
                df = pd.read_csv(file_path)

                # Replace NaN values with an empty string
                df['message'] = df['message'].fillna('')

                # Use TF-IDF to transform the text data
                vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english')
                tfidf = vectorizer.fit_transform(df['message'])

                # Apply NMF to the TF-IDF matrix with increased max_iter
                nmf = NMF(n_components=n_topics, random_state=1, max_iter=500).fit(tfidf)

                # Get the topic distribution for each document
                topic_dist = nmf.transform(tfidf)

                # Get the top words for each topic
                tfidf_feature_names = vectorizer.get_feature_names_out()
                topics = print_top_words(nmf, tfidf_feature_names, n_top_words)

                # Write the topics, their top words and their average scores to the results DataFrame
                for i, topic in enumerate(topics):
                    results_df.loc[len(results_df)] = [file_path, i + 1, topic, topic_dist[:, i].mean()]

    # Write the results DataFrame to a CSV file
    results_df.to_csv(output_file, index=False)


# Use the function
nmf_topic_modeling("../clustering_keyword_search", "topic_modeling_results_nmf_gg.csv", 5, 10)
