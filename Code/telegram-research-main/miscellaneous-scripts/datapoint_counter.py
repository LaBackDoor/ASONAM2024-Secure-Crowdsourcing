import pandas as pd

# Read the csv
df = pd.read_csv('combined_csv_data.csv')

# Drop rows with empty date
df = df.dropna(subset=['date'])

# Convert the date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract the year from the date
df['year'] = df['date'].dt.year

# Count the number of messages per year
final_table = df.groupby('year').size().reset_index(name='message_count')

print(final_table)
