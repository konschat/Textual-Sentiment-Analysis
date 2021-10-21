# Textual-Sentiment-Analysis

This repository involves opinion mining, specifically textual sentiment analysis of patient's clinical reviews based on their medical experience. In Textual_Analysis.py, EDA (with visualizations included) is performed while the performance of heterogenous classifiers is examined. VADER (Valence Aware Dictionary and sEntiment Reasoner) [2] is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media or texts from other domains, which is used in order to produce each review's sentiment. The final stage of the repository examines performance fluctuations of each classifier based on hyper-parameter tuning such as tolerance, alphas and number of iterations. The initial data [1] can be found at : https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29 

Acknowledgements - Citations 

[1]. Felix Gräßer, Surya Kallumadi, Hagen Malberg, and Sebastian Zaunseder. 2018. Aspect-Based Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning. In Proceedings of the 2018 International Conference on Digital Health (DH '18). ACM, New York, NY, USA, 121-125.

[2]. Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
