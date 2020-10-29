import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import json_normalize
from pymongo import MongoClient, errors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os
from pathlib import Path
from string import punctuation
from bs4 import BeautifulSoup
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim import corpora
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import guidedlda
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation as LDA

 
 
def load_twitter_df_from_file(filename):
    # directory = "../data/"
    # file_path = directory+filename
    with open(filename) as data_file:
        data = json.load(data_file)
    # data_df = json_normalize(data)
    data_df = pd.DataFrame(data, columns=['id_str', 'text', 'lang'])
    return data_df


def remove_punctuation(string, punctuation):
    # remove given punctuation marks from a string
    for character in punctuation:
        string = string.replace(character, '')
    return string


def lemmatize_str(string):
    # Lemmatize a string and return it in its original format
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w)
                    for w in w_tokenizer.tokenize(string)
                    if "http" not in w])


def clean_column(df, column):
    # Apply data cleaning pipeline to a given pandas DataFrame column
    df[column] = df[column].apply(lambda x: str(x).lower())
    df[column] = df[column].apply(lambda x: remove_punctuation(x, punctuation))
    df[column] = df[column].apply(lambda x: lemmatize_str(x))
    return


def get_stop_words(new_stop_words=None):
    # Retrieve stop words and append any additional stop words
    stop_words = list(STOPWORDS)
    if new_stop_words:
        stop_words.extend(new_stop_words)
    return set(stop_words)


def vectorize(df, column, stop_words):
    # Vectorize a text column of a pandas DataFrame
    text = df[column].values
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(text)
    features = np.array(vectorizer.get_feature_names())
    return X, features


def plot_10_most_common_words(count_data, count_vectorizer):
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("../images/common_words.png")
    # plt.show()


if __name__ == "__main__":

    stopwords = set(STOPWORDS)
    stopwords.update(["im", "nike", "check", "out", "rt", "air", "know", "hey",
                     "httpstcoyyv8xsbp4x", "good", "share", "keep", "new",
                      "shoe", "im  loving", "item", "sneaker", "let", "see",
                      "got", "weight", "jordan", "loving  on", "on"])

    plt.rcParams.update({'font.size': 16})
    punc = punctuation

    # directory = "../data/"
    # df_list = []
    # for filename in os.listdir(directory):
    #     if filename.endswith(".json"):
    #         df_list.append(load_twitter_df_from_file("../data/"+filename))
    #         print("../data/"+filename)
    #         continue
    #     else:
    #         continue

    all_df = pd.read_json('../data/concatenated_abridged.jsonl', lines = True)

    # all_df = pd.concat(df_list)
    print(all_df.head())
    print(all_df.tail())
    print(all_df.shape)

    clean_column(all_df, 'full_text')
    print(all_df.head())

    english_df = all_df["lang"] == "en"
    print(english_df.head())

    all_df = all_df[english_df]
    print(all_df.head())
    # directory = '../data'
    # # df_namer = "df"
    # counter = 1

    # print(my_dict)

    # Commented out
    # TF IDF
    v = TfidfVectorizer()  # max_features = 30000
    X = v.fit_transform(all_df['full_text'])

    # Load the library with the CountVectorizer method

    sns.set_style('whitegrid')
    # Helper function

    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer(stop_words=stopwords)
    # Fit and transform the processed titles
    count_data = count_vectorizer.fit_transform(all_df['full_text'])
    # Visualise the 10 most common words
    plot_10_most_common_words(count_data, count_vectorizer)

    text = " ".join(tweet for tweet in all_df.full_text)
    print("There are {} words in the combination of all review.".format(len(text)))

    wordcloud = WordCloud(stopwords=stopwords).generate(text)
    fig, ax = plt.subplots()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("../images/wordcloud.png")
    # plt.show()

    #  Coronavirus, Equality/Diversity, climate/enviromental,
    # business/finance, and other as a catachall

    vocab = count_vectorizer.get_feature_names()
    word2id = dict((v, idx) for idx, v in enumerate(vocab))
    # print(vocab)

    seed_topic_list = [['coronavirus', 'covid', 'covid-19', 'virus', 'curve',
                        'flat', 'coronavirusoutbreak', 'corona'],
                       ['equality', 'diversity', 'china', 'equalitywarrior',
                        'equally', 'equal', 'diversityandinclusion',
                        'diverse', 'chinese', 'uyghur', 'forceduyghurlabor',
                        'forced', 'forcedlabour', 'forcedlabor',
                        'forcedchildlabor', 'forcedslave'],
                       ['climate', 'environmental', 'climatechange', 'fuel',
                        'fossil', 'renewable', 'carbon'],
                       ['business', 'finance', 'economic',
                        'investment', 'money',
                        'bank', 'stocks', 'stockstobuy', 'stockstowatch',
                        'stockmarkets'],
                       ['fashion', 'poshmarkapp', 'poshmark', 'style']]
    model = guidedlda.GuidedLDA(n_topics=7, n_iter=100, random_state=7, refresh=20)

    seed_topics = {}
    for t_id, st in enumerate(seed_topic_list):
        for word in st:
            if word in word2id:
                seed_topics[word2id[word]] = t_id

    model.fit(count_data, seed_topics=seed_topics, seed_confidence=0.95)

    topic_word = model.topic_word_
    n_top_words = 10
    topics = ['coronavirus', 'equality/diversity', 'climate',
              'business/finance', 'other', 'other2', 'other3']
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        print(topics[i])
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    doc_topic = model.transform(count_data)
    for i in range(9):
        print("top topic: {} Document: {}".format(doc_topic[i].argmax(),
                                                  ', '.join(np.array(vocab)[count_data[i, :].toarray().argsort()][0][::-1][0:10])))



