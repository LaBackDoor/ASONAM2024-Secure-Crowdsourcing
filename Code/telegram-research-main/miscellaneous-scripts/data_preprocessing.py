import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re


def get_page_title(url):
    try:
        response = requests.get(url, timeout=3)
        response.raise_for_status()  # raise HTTPError for bad status codes
    except (requests.HTTPError, requests.RequestException) as e:
        print(f'Error: {e}, URL: {url}')
        return None
    else:
        soup = BeautifulSoup(response.text, 'html.parser')
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
            print(f'Successfully got title: "{title}", URL: {url}')
            return title
        else:
            print(f'No title found, URL: {url}')
            return None


def replace_links_with_titles(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            print(f'Processing file: {filename}')
            df = pd.read_csv(os.path.join(
                directory, filename), low_memory=False)

            # Assume that 'message' column exists
            assert 'message' in df.columns

            # Make a copy to not change data while iterating
            df_copy = df.copy()

            url_pattern = re.compile(
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+:/]|[\w+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

            # Counters
            success_count = 0
            fail_count = 0
            drop_count = 0

            for i, row in df.iterrows():
                message = row['message']
                if not isinstance(message, str):
                    df_copy = df_copy.drop(i)
                    drop_count += 1
                    continue
                urls = url_pattern.findall(message)

                for url in urls:
                    title = get_page_title(url)

                    # If get_page_title returned None, a 400s error was returned
                    # or no title found so we delete this row in the copy
                    if title is None:
                        df_copy = df_copy.drop(i)
                        drop_count += 1
                        fail_count += 1
                        break  # move to the next row
                    else:
                        # Replace the link in the message with the title
                        df_copy.at[i, 'message'] = message.replace(url, title)
                        success_count += 1

            # Make Directory if Doesn't Exist
            output_dir = "preprocessed_file"

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Write the copy to a new CSV file
            df_copy.to_csv(os.path.join(
                output_dir, "preprocessed_" + filename), index=False)

            print(f'Finished processing file: {filename}')
            print(f'Successful requests: {success_count}')
            print(f'Failed requests: {fail_count}')
            print(f'Dropped rows: {drop_count}')


replace_links_with_titles('clustering_keyword_search')
