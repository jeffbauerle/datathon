import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":

    df = pd.read_pickle('../data/new_features.pkl')
    
    plt.figure(figsize=(20, 12))
    plt.subplot(111)
    arr1 = plt.scatter(df.created_at, df.favorite_count, c=df.vader_compound)
    plt.xlim(['2020-09-30 00:00:00+00:00','2020-10-02 23:59:54+0000'])
    plt.ylim([0, 1000])
    plt.title('Tweets by Favorites Over Time', fontsize = 25)
    plt.ylabel('Favorites', fontsize = 18)
    plt.xlabel('Date', fontsize = 18)
    plt.yscale('linear')
    bar = plt.colorbar()
    bar.set_label('Compound Vader Sentiment', rotation=270)
    plt.grid(True)
    plt.savefig('../images/all_tweets_by_favs')

    plt.figure(figsize=(20, 12))
    plt.subplot(111)
    arr1 = plt.scatter(df.created_at, df.retweet_count, c=df.vader_compound)
    plt.xlim(['2020-09-30 00:00:00+00:00','2020-10-02 23:59:54+0000'])
    plt.ylim([0, 60000])
    plt.title('Retweets Over Time', fontsize = 25)
    plt.ylabel('Retweets', fontsize = 18)
    plt.xlabel('Date', fontsize = 18)
    plt.yscale('linear')
    bar = plt.colorbar()
    bar.set_label('Compound Vader Sentiment', rotation=270)
    plt.grid(True)
    plt.savefig('../images/all_tweets_by_rt')



    
