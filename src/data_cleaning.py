import pandas as pd
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text_str):
    vs = analyzer.polarity_scores(text_str)
    return vs

def vader_neg(text_str):
    vs = analyzer.polarity_scores(text_str)
    return vs['neg']

def vader_neu(text_str):
    vs = analyzer.polarity_scores(text_str)
    return vs['neu']

def vader_pos(text_str):
    vs = analyzer.polarity_scores(text_str)
    return vs['pos']
    
def vader_compound(text_str):
    vs = analyzer.polarity_scores(text_str)
    return vs['compound']

def detect_trump(string):
    lower = string.lower()
    split = lower.split()
    for word in split:
        if 'trump' in split:
            return True
        if '#trump' in split:
            return True
    else:
        return False

def detect_biden(string):
    lower = string.lower()
    split = lower.split()
    for word in split:
        if 'biden' in split:
            return True
        if '#biden' in split:
            return True
    else:
        return False

def detect_obama(string):
    lower = string.lower()
    split = lower.split()
    for word in split:
        if 'obama' in split:
            return True
        if '#obama' in split:
            return True
    else:
        return False

def detect_clinton(string):
    lower = string.lower()
    split = lower.split()
    for word in split:
        if 'clinton' in split:
            return True
        if '#clinton' in split:
            return True
        if 'hillary' in split:
            return True
        if '#hillary' in split:
            return True
    else:
        return False

def detect_bush(string):
    lower = string.lower()
    split = lower.split()
    for word in split:
        if 'bush' in split:
            return True
        if '#bush' in split:
            return True
    else:
        return False

def get_handle(d):
    return d['screen_name']

def get_followers(d):
    return d['followers_count']

def get_friends(d):
    return d['friends_count']

def get_statuses(d):
    return d['statuses_count']

def get_creation_date(d):
    return d['created_at']
    
def get_location(d):
    return d['location']

def get_acct_favs(d):
    return d['favourites_count']

def suspect_name(name):
    return name[-5:].isnumeric()


if __name__ == '__main__':
    df = pd.read_json('../data/concatenated_abridged.jsonl', lines = True)
    analyzer = SentimentIntensityAnalyzer()
    df['vader_neg'] = df['full_text'].apply(vader_neg)
    df['vader_pos'] = df['full_text'].apply(vader_pos)
    df['vader_neu'] = df['full_text'].apply(vader_neu)
    df['vader_compound'] = df['full_text'].apply(vader_compound)
    df['trump'] = df['full_text'].apply(detect_trump)
    df['biden'] = df['full_text'].apply(detect_biden)
    df['obama'] = df['full_text'].apply(detect_obama)
    df['clinton'] = df['full_text'].apply(detect_clinton)
    df['bush'] = df['full_text'].apply(detect_bush)
    df['friends'] = df['user'].apply(get_friends)
    df['status count'] = df['user'].apply(get_statuses)
    df['followers'] = df['user'].apply(get_followers)
    df['creation date'] = df['user'].apply(get_creation_date)
    df['handle'] = df['user'].apply(get_handle)
    df['location'] = df['user'].apply(get_location)
    df['acct favs'] = df['user'].apply(get_acct_favs)
    df['trailingnumbers'] = df['handle'].apply(suspect_name)

    df.to_pickle('../data/new_features.pkl')





