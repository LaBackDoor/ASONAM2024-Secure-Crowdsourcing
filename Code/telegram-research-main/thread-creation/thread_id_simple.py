import os
import pandas as pd


def create_directory(directory_path):
    """
    Create a new directory if it doesn't exist.

    :param directory_path: The path of the directory to be created.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def get_unique_ids(dataframe, column):
    """
    Get unique IDs from a DataFrame's column.

    :param dataframe: The DataFrame to search.
    :param column: The column in the DataFrame to find unique IDs.
    :return: A list of unique IDs.
    """
    return dataframe[column].unique()


def create_cluster(dataframe, reply_column, id_column, unique_id):
    """
    Create a cluster by finding rows that match the unique_id in the reply_column and id_column.

    :param dataframe: The DataFrame to search.
    :param reply_column: The reply column name in the DataFrame.
    :param id_column: The id column name in the DataFrame.
    :param unique_id: The unique_id to search for.
    :return: A new DataFrame that forms the cluster.
    """
    return dataframe[(dataframe[reply_column] == unique_id) | (dataframe[id_column] == unique_id)]


def process_files(search_directory, ground_truth_directory):
    """
    Process each CSV file in the search directory, creating clusters and saving them to the ground truth directory.

    :param search_directory: The directory to search for CSV files.
    :param ground_truth_directory: The directory to save the cluster CSV files.
    """
    # Go through each CSV file in the search directory
    for filename in os.listdir(search_directory):
        if filename.endswith('.csv'):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(os.path.join(search_directory, filename), low_memory=False)

            # Create a subdirectory in ground_truth_directory for this CSV file
            subdirectory = os.path.join(ground_truth_directory, filename[:-4])
            create_directory(subdirectory)

            # Find unique ids in the 'reply_to_msg_id' column
            unique_ids = get_unique_ids(df, 'reply_to_msg_id')

            # For each unique id, create a new DataFrame with rows that have the same id
            for unique_id in unique_ids:
                df_cluster = create_cluster(df, 'reply_to_msg_id', 'id', unique_id)

                # Only save this DataFrame to a new CSV file in the subdirectory if it has more than one row
                if len(df_cluster) > 1:
                    df_cluster.to_csv(os.path.join(subdirectory, f'{unique_id}.csv'), index=False)


# Specify your directories
search_directory = 'clustering_keyword_search'
ground_truth_directory = 'ground_truth'

# Check if ground_truth_directory exists, if not, create it
create_directory(ground_truth_directory)

# Process files in the search directory
process_files(search_directory, ground_truth_directory)
