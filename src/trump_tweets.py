import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text_str):
    vs = analyzer.polarity_scores(text_str)
    return vs

def vader_compound(text_str):
    vs = analyzer.polarity_scores(text_str)
    return vs['compound']

def vader_neg(text_str):
    vs = analyzer.polarity_scores(text_str)
    return vs['neg']

def vader_neu(text_str):
    vs = analyzer.polarity_scores(text_str)
    return vs['neu']

def vader_pos(text_str):
    vs = analyzer.polarity_scores(text_str)
    return vs['pos']
    

if __name__ == '__main__':
    trump_df = pd.read_csv('../data/trumptweets.csv')

    analyzer = SentimentIntensityAnalyzer()

    trump_df['vader_comp'] = trump_df['text'].apply(vader_compound)
    trump_df['vader_neg'] = trump_df['text'].apply(vader_neg)
    trump_df['vader_pos'] = trump_df['text'].apply(vader_pos)
    trump_df['vader_neu'] = trump_df['text'].apply(vader_neu)
    trump_df['datetime'] = pd.to_datetime(trump_df['date'])

    del_df = trump_df[trump_df['isDeleted'] != 'f']
    weekly_df_mean = trump_df.set_index('datetime').groupby(pd.Grouper(freq='W')).mean()
    weekly_df_med = trump_df.set_index('datetime').groupby(pd.Grouper(freq='W')).median()
    weekly_df_sum = trump_df.set_index('datetime').groupby(pd.Grouper(freq='W')).sum()

#generating matplotlib images of trump's happiness

    fig, ax = plt.subplots(figsize=(20, 12))
    ax.scatter(trump_df.datetime, trump_df.vader_comp)
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Compound Sentiment', fontsize=18)
    ax.set_title('My Trump Mood Board: Compound Sentiment (All Tweets)', fontsize=25)
    ax.grid(True)
    plt.savefig('../images/compiled_trump_mood.png')

    fig, ax = plt.subplots(figsize=(20, 12))
    ax.scatter(trump_df.datetime, trump_df.vader_neg, )
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Vader Negative Sentiment', fontsize=18)
    ax.set_title('My Trump Mood Board – Tweets by Negative Sentiment', fontsize=25)
    ax.grid(True)
    ax.legend(loc='upper left');
    plt.savefig('../images/negative_trump_mood.png')

    fig, ax = plt.subplots(figsize=(20, 12))
    ax.scatter(trump_df.datetime, trump_df.vader_pos)
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Vader Positive Sentiment', fontsize=18)
    ax.set_title('My Trump Mood Board – Tweets by Positive Sentiment', fontsize=25)
    ax.grid(True)
    ax.legend(loc='upper left');
    plt.savefig('../images/positive_trump_mood.png')

    fig, ax = plt.subplots(figsize=(20, 12))
    ax.hist(trump_df.datetime, bins = 4191)
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Tweets', fontsize=18)
    ax.set_title('How much is he tweeting? – Daily', fontsize=25)
    ax.grid(True)
    plt.savefig('../images/daily_tweet_count.png')

    fig, ax = plt.subplots(figsize=(20, 12))
    ax.hist(trump_df.datetime, bins = 598)
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Tweets', fontsize=18)
    ax.set_title('How much is he tweeting? – Weekly', fontsize=25)
    ax.grid(True)
    plt.savefig('../images/weekly_tweet_count.png')

    fig, ax = plt.subplots(figsize=(20, 12))
    ax.hist(del_df.datetime, bins = 137)
    ax.set_xlabel('Date', fontsize= 18)
    ax.set_ylabel('Tweets (Count)', fontsize= 18)
    ax.set_title('Deleted Trump Tweets – Weekly', fontsize= 25)
    ax.grid(True)
    plt.savefig('../images/weekly_del_tweet_count.png')

    fig, ax = plt.subplots(figsize=(20, 12))
    ax.scatter(weekly_df_mean.index, weekly_df_mean.vader_comp, c=weekly_df_mean.vader_comp, s=200)
    ax.set_xlabel('Date', fontsize = 18)
    ax.set_ylabel('Vader Compound Sentiment', fontsize = 18)
    ax.set_title('My Trump Mood Board – Weekly Mean Vader Compound Sentiment', fontsize=25)
    ax.grid(True)
    plt.savefig('../images/weekly_mean_comp_sent.png')

    fig, ax = plt.subplots(figsize=(20, 12))
    ax.scatter(weekly_df_sum.index, weekly_df_sum.favorites, c=weekly_df_mean.vader_comp, s=200)
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Favorites, Total', fontsize=18)
    ax.set_title('My Trump Mood Board – Weekly Favorites, Total', fontsize=25)
    ax.grid(True)
    plt.savefig('../images/weekly_sum_favs_total.png')

    fig, ax = plt.subplots(figsize=(20, 12))
    ax.scatter(weekly_df_sum.index, weekly_df_sum.retweets, c=weekly_df_mean.vader_comp, s=200)
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Retweets, Total', fontsize=18)
    ax.set_title('My Trump Mood Board – Weekly Retweets, Total', fontsize=25)
    ax.grid(True)
    plt.savefig('../images/weekly_sum_retweets_total.png')

    fig, ax = plt.subplots(figsize=(20, 12))
    ax.scatter(weekly_df_mean.index, weekly_df_mean.favorites, c=weekly_df_mean.vader_comp, s=200)
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Average favorites per tweet', fontsize=18)
    ax.set_title('My Trump Mood Board – Weekly Retweets, Mean', fontsize=25)
    ax.grid(True)
    plt.savefig('../images/xx_weekly_favorites_mean.png')

    plt.rc('font', size=12)
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.scatter(weekly_df_mean.index, weekly_df_mean.retweets, c=weekly_df_mean.vader_comp, s=200)
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Average Weekly Retweets', fontsize=18)
    ax.set_title('My Trump Mood Board – Weekly Retweets, Mean', fontsize=25)
    ax.grid(True)
    plt.savefig('../images/weekly_retweets_mean.png')












