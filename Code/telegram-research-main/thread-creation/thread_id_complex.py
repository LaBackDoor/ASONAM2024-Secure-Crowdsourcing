import os
import pandas as pd
import numpy as np


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def create_cluster(dataframe, reply_column, id_column, unique_id):
    # Initialize the cluster DataFrame
    cluster = dataframe[dataframe[id_column] == unique_id]

    while True:
        # Find messages that are being replied to
        replied_to = dataframe[dataframe[id_column].isin(
            cluster[reply_column])]

        # Find replies to the messages currently in the cluster
        replies = dataframe[dataframe[reply_column].isin(cluster[id_column])]

        # Concatenate these dataframes
        new_cluster_data = pd.concat([replied_to, replies]).drop_duplicates()

        # Add the replied messages that are not in the ID column
        not_in_id = dataframe[dataframe[reply_column].isin(cluster[id_column]) &
                              ~dataframe[id_column].isin(cluster[id_column])]
        new_cluster_data = pd.concat(
            [new_cluster_data, not_in_id]).drop_duplicates()

        # If there are no new messages to be added, break
        if new_cluster_data.empty or new_cluster_data[id_column].isin(cluster[id_column]).all():
            break

        # Otherwise, add the new messages to the cluster and repeat
        cluster = pd.concat([cluster, new_cluster_data]).drop_duplicates()

    return cluster


def process_files(search_directory, ground_truth_directory):
    for filename in os.listdir(search_directory):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(
                search_directory, filename), low_memory=False)
            subdirectory = os.path.join(ground_truth_directory, filename[:-4])
            create_directory(subdirectory)

            # Sorted unique ids in the 'id' and 'reply_to_msg_id' columns
            unique_ids = np.sort(
                pd.concat([df['id'], df['reply_to_msg_id']]).unique())

            for unique_id in unique_ids:
                df_cluster = create_cluster(
                    df, 'reply_to_msg_id', 'id', unique_id)

                if len(df_cluster) > 1:
                    df_cluster.sort_values(by=['id']).to_csv(
                        os.path.join(subdirectory, f'{unique_id}.csv'), index=False)

                    # Drop used messages from the dataframe
                    df = df.drop(df_cluster.index)


# Specify your directories
search_directory = 'clustering_keyword_search'
ground_truth_directory = 'ground_truth_complex'

create_directory(ground_truth_directory)

process_files(search_directory, ground_truth_directory)
