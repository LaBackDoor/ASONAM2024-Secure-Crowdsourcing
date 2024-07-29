import os
import pandas as pd
import nltk
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.add('http')  # Add 'http' to stop words
tokenizer = RegexpTokenizer(r'\w+')  # To remove punctuation


def preprocess_text(text):
    tokens = tokenizer.tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if
              token not in stop_words and token.isalpha() and 'http' not in token]
    return tokens


def lda_topic_modeling(input_folder, output_filename, n_topics, n_top_words):
    dfs = []

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(root, file))

                documents = []

                # Replace NaN values with an empty string
                df['message'] = df['message'].fillna('')

                for message in df['message']:
                    documents.append(preprocess_text(message))

                dictionary = corpora.Dictionary(documents)
                corpus = [dictionary.doc2bow(doc) for doc in documents]

                lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics)

                topics = []
                for i in range(n_topics):
                    words = lda_model.show_topic(i, topn=n_top_words)
                    topics.append(", ".join([word for word, prop in words]))

                coherence_model = CoherenceModel(model=lda_model, texts=documents, dictionary=dictionary,
                                                 coherence='c_v', processes=1)
                coherence_score = coherence_model.get_coherence()

                topics_df = pd.DataFrame({
                    'filename': [os.path.join(root, file)] * n_topics,
                    'topic': range(1, n_topics + 1),
                    'top_words': topics,
                    'coherence_score': [coherence_score] * n_topics
                })

                dfs.append(topics_df)

    result_df = pd.concat(dfs, ignore_index=True)
    result_df.to_csv(output_filename, index=False)


lda_topic_modeling("../clustering_keyword_search", "topic_modeling_results_nltk_gg.csv", 5, 3)
