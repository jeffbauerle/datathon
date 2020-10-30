import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy
import numpy as np
from statsmodels.stats.multitest import multipletests

if __name__ == "__main__":
    pos_t = pd.read_pickle('../data/trump_pos.pkl')
    neg_t = pd.read_pickle('../data/trump_neg.pkl')
    pos_b = pd.read_pickle('../data/biden_pos.pkl')
    neg_b = pd.read_pickle('../data/biden_neg.pkl')
    
    #print(pos_t.info())
    def make_lst_for_box_plot(df):
        lst = df['retweet_count'].tolist()
        return lst 
    
    pos_t_lst = make_lst_for_box_plot(pos_t)
    neg_t_lst = make_lst_for_box_plot(neg_t)
    pos_b_lst = make_lst_for_box_plot(pos_b)
    neg_b_lst = make_lst_for_box_plot(neg_b)

    data = [pos_t_lst, neg_t_lst, pos_b_lst, neg_b_lst]
    labels = ["Positive & @Trump", "Negative & @Trump", "Positive & @Biden", "Negative & @Biden"]
    meanpointprops = dict(marker='D', markeredgecolor='black',
                      markerfacecolor='firebrick')
    fig, ax = plt.subplots(figsize=(10,8))
    ax.boxplot(data, labels=labels, meanprops=meanpointprops, showmeans=True)
    ax.set_title("Sentiment Analysis and Mentions")
    ax.set_xlabel("Sentiment Analysis and Mentions")
    ax.set_ylabel("Number of Retweets")
    plt.tight_layout()
    plt.savefig('../images/sentiment_analysis_box_plot')
    #plt.show()

    #Kruskal–Wallis Test due distribution not being normal
    F, p = stats.f_oneway(pos_t_lst, neg_t_lst, pos_b_lst, neg_b_lst)
    print(F)
    print(p)
    # p-value smaller than our alpha of 0.05 (6.6*10-110)
    
    #MannWhitneyU test
    #t t
    U, p1 = scipy.stats.mannwhitneyu(pos_t_lst, neg_t_lst)
    #b b
    U, p2 = scipy.stats.mannwhitneyu(pos_b_lst, neg_b_lst)
    #+t +b
    U, p3 = scipy.stats.mannwhitneyu(pos_t_lst, pos_b_lst)
    #-t -b
    U, p4 = scipy.stats.mannwhitneyu(neg_t_lst, neg_b_lst)
    #+t -b
    U, p5 = scipy.stats.mannwhitneyu(pos_t_lst, neg_b_lst)
    #-t +b
    U, p6 = scipy.stats.mannwhitneyu(neg_t_lst, pos_b_lst)

    bonf_correction = multipletests([p, p1, p2, p3, p4, p5, p6],
                    alpha=0.05, method='bonferroni')
    print(bonf_correction)
