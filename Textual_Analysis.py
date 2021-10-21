import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import style
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics.pairwise import chi2_kernel, additive_chi2_kernel, euclidean_distances
style.use('ggplot')
import nltk
from nltk.corpus import stopwords
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold  # import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from  sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import seaborn as sn

test = pd.read_csv('drugsComTest_raw.csv', error_bad_lines=False)
train = pd.read_csv('drugsComTrain_raw.csv', error_bad_lines=False)

print(train.head(10))
print(test.head(10))

print(f'Train has {train.shape[0]} number of rows and {train.shape[1]} number of columns')
print(f'Test has {test.shape[0]} number of rows and {test.shape[1]} number of columns')

df = pd.concat([train,test])
print("Required shape check", train.shape, test.shape, df.shape)
print("Merged dataset head : \n", df.head(10))

# Correlation matrix for merged dataframe(shows only numerical values)
corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()
                ################
                # Start of EDA #
                ################

print(df.describe(include='all'))
print(df.isnull().sum()/df.shape[0])

df.columns = ['Id','drugName','condition','review','rating','date','usefulCount']
print("New clumns check : \n", df.head)

#check number of unique values in drugName
print(df['drugName'].nunique())
#check number of unique values in condition
print(df['condition'].nunique())


# # Date conversion to datetime
# df['date'] = pd.to_evaluate(df['date'])

# Create new dataset with review and rating for sentiment analysis purposes
df2 = df[['Id', 'review', 'rating']].copy()
print("New dataset head : \n", df2.head(10))

# Null check
print(df2.isnull().any().any())
# Check dataset datatype, also for null
print(df2.info())
# Show Unique Id as array, Id count and unique Id values
print(df2['Id'].unique(), df2['Id'].count(), df2['Id'].nunique())

# Access individual value
print(df['review'][1])
nltk.download(['punkt', 'stopwords'])

stopwords = stopwords.words('english')
print(df2.head(10))

# Remove stopwords from review
df2['cleanReview'] = df2['review'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))
print(df2.head(10))


            #####################################################
            # Some visualizations to better understand the data #
            #####################################################

# Plot a bar-graph to check top 20 conditions
plt.figure(figsize=(12,6))
conditions = df['condition'].value_counts(ascending = False).head(20)

plt.bar(conditions.index,conditions.values)
plt.title('Top-20 Conditions',fontsize = 20)
plt.xticks(rotation=90)
plt.ylabel('Count')
plt.show()

# Plot a bar-graph to check bottom 20 conditions
plt.figure(figsize=(12,6))
conditions_bottom = df['condition'].value_counts(ascending = False).tail(20)

plt.bar(conditions_bottom.index,conditions_bottom.values)
plt.title('Bottom-20 Conditions',fontsize = 20)
plt.xticks(rotation=90)
plt.ylabel('Count')
plt.show()

# Plot a bar-graph to check top 20 drugName
plt.figure(figsize=(12,6))
drugName_top = df['drugName'].value_counts(ascending = False).head(20)

plt.bar(drugName_top.index,drugName_top.values,color='red')
plt.title('drugName Top-20',fontsize = 20)
plt.xticks(rotation=90)
plt.ylabel('Count')
plt.show()

# Plot a bar-graph to check bottom 20 drugName
plt.figure(figsize=(12,6))
drugName_bottom = df['drugName'].value_counts(ascending = False).tail(20)

plt.bar(drugName_bottom.index,drugName_bottom.values,color='red')
plt.title('drugName Bottom-20',fontsize = 20)
plt.xticks(rotation=90)
plt.ylabel('count')
plt.show()

# Checking ratings distribution
ratings_ = df['rating'].value_counts().sort_values(ascending=False).reset_index().\
                    rename(columns = {'index' :'rating', 'rating' : 'counts'})
ratings_['percent'] = 100 * (ratings_['counts']/df.shape[0])
print(ratings_)

# Setting the Parameters
sns.set(font_scale = 1.2, style = 'dark')
plt.rcParams['figure.figsize'] = [12, 6]

# Plot and check
sns.barplot(x = ratings_['rating'], y = ratings_['percent'],order = ratings_['rating'])
plt.title('Ratings Percent',fontsize=20)
plt.show()

# Plot a distplot of usefulCount between critics
sns.distplot(df['usefulCount'])
plt.show()

# Check the descriptive summary
sns.boxplot(y = df['usefulCount'])
plt.show()

# Check if a single drug is used for multiple conditions
drug_multiple_cond = df.groupby('drugName')['condition'].nunique().sort_values(ascending=False)
print(drug_multiple_cond.head(10))

# Check the number of drugs in the dataset condition wise
# Also plot the top 20
conditions_gp = df.groupby('condition')['drugName'].nunique().sort_values(ascending=False)

condition_gp_top_20 = conditions_gp.head(20)
sns.set(font_scale = 1.2, style = 'darkgrid')
plt.rcParams['figure.figsize'] = [12, 6]
sns.barplot(x = condition_gp_top_20.index, y = condition_gp_top_20.values)
plt.title('Top-20 Number of drugs per condition',fontsize=20)
plt.xticks(rotation=90)
plt.ylabel('count',fontsize=10)
plt.show()


            ###############################
            # Start of sentiment analysis #
            ###############################

analyzer = SentimentIntensityAnalyzer()

df2['vaderReviewScore'] = df2['cleanReview'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
df2.head(10)

positive_num = len(df2[df2['vaderReviewScore'] >=0.05])
neutral_num = len(df2[(df2['vaderReviewScore'] >-0.05) & (df2['vaderReviewScore']<0.05)])
negative_num = len(df2[df2['vaderReviewScore']<=-0.05])

print("Positive,Neutral,Negative : ", positive_num,neutral_num, negative_num)

# Toral vadef Sentiment = positive + negative + neutral
df2['vaderSentiment']= df2['vaderReviewScore'].map(lambda x:int(2) if x>=0.05 else int(1) if x<=-0.05 else int(0))
print("Vader Sentiment value counts : \n", df2['vaderSentiment'].value_counts())

df2.loc[df2['vaderReviewScore'] >=0.05,"vaderSentimentLabel"] ="positive"
df2.loc[(df2['vaderReviewScore'] >-0.05) & (df2['vaderReviewScore']<0.05),"vaderSentimentLabel"]= "neutral"
df2.loc[df2['vaderReviewScore']<=-0.05,"vaderSentimentLabel"] = "negative"

print(df2.shape)
print(df2.head(10))

positive_rating = len(df2[df2['rating'] >=7.0])
neutral_rating = len(df2[(df2['rating'] >=4) & (df2['rating']<7)])
negative_rating = len(df2[df2['rating']<=3])

print("Positive,Neutral,Negative : ", positive_rating,neutral_rating,negative_rating)

df2['ratingSentiment']= df2['rating'].map(lambda x:int(2) if x>=7 else int(1) if x<=3 else int(0) )
print(df2['ratingSentiment'].value_counts())

df2.loc[df2['rating'] >=7.0,"ratingSentimentLabel"] ="positive"
df2.loc[(df2['rating'] >=4.0) & (df2['rating']<7.0),"ratingSentimentLabel"]= "neutral"
df2.loc[df2['rating']<=3.0,"ratingSentimentLabel"] = "negative"

df2 = df2[['Id','review','cleanReview','rating','ratingSentiment','ratingSentimentLabel','vaderReviewScore','vaderSentiment','vaderSentimentLabel']]
print(df2.head(10))

# Correlation matrix for df2 dataframe(only numerical values)
corrMatrix = df2.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

            #######################################
            # Model-data build sentiment analysis #
            #######################################

# print("Columns check")
# print(df2['vaderReviewScore'])
# print(df2['vaderSentiment'])
# print(df2['ratingSentiment'])
# print(df2['ratingSentimentLabel'])
# print("Done")

# Create the TF-IDF vectorizer and transforms the corpus
vectorizer = TfidfVectorizer()
reviews_corpus = vectorizer.fit_transform(df.review)

print(reviews_corpus)
print(reviews_corpus.shape)

# Dependent feature
sentiment = df2['vaderSentiment']
print(sentiment.head(10))
print(sentiment.shape)

rating = df2['ratingSentiment']
print(rating.head(10))
print(rating.shape)

# Split the data in train and test
X_train,X_test,Y_train,Y_test = train_test_split(reviews_corpus,sentiment,test_size=0.33,random_state=42, shuffle=True)

print('Train data shape ',X_train.shape,Y_train.shape)
print('Test data shape ',X_test.shape,Y_test.shape)

C = [0.01, 0.05, 0.25, 0.5, 1]
alphas  = [0.01, 0.1, 0.5, 1]
iters = [10, 100, 1000, 5000, 10000]

# from sklearn.impute import SimpleImputer
# import numpy as np

# imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
# revied_corpus_imp = imp_mean.fit_transform(reviews_corpus)

# skf = StratifiedKFold()
# for train_index, test_index in skf.split(reviews_corpus, sentiment):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = reviews_corpus[train_index], reviews_corpus[test_index]
#     Y_train, Y_test = sentiment[train_index], sentiment[test_index]

    # X_train = pd.isnull(X_train)
    # X_train = pd.lreshape(X_train,dropna=True)
    # X_test = pd.isnull(X_test)
    # X_test = pd.lreshape(X_test, dropna=True)

for a in alphas:
    # Fit the Multinomial model and predict the output
    clf = MultinomialNB(alpha=a).fit(X_train, Y_train)

    multinb_pred = clf.predict(X_test)

    print("Multinomial Accuracy: %s" % str(clf.score(X_test, Y_test)))
    print("Multinomial F1-score: %s" % str(f1_score(Y_test, multinb_pred, average='macro', zero_division=1)))
    print("Confusion Matrix")
    print(confusion_matrix(multinb_pred, Y_test))

    print(classification_report(Y_test,multinb_pred,target_names= df2['vaderSentimentLabel'].unique()))

    plot_confusion_matrix(clf, X_test, Y_test)
    plt.show()


for k in range(0,len(iters)):
    # saga solver supported for the l1 penalty
    clf = LogisticRegression(penalty='l2', tol=0.001, C=1.0, fit_intercept=True, intercept_scaling=1,
                           solver='lbfgs', max_iter=iters[k], multi_class='auto',
                        warm_start=False, l1_ratio=None).fit(X_train, Y_train)

    logreg_pred = clf.predict(X_test)

    print("Logistic Regression Accuracy: %s" % str(clf.score(X_test, Y_test)))
    print("Logistic Regression F1-score: %s" % str(f1_score(Y_test, logreg_pred, average='macro', zero_division=1)))
    print("Confusion Matrix")
    print(confusion_matrix(logreg_pred, Y_test))

print(classification_report(Y_test,logreg_pred,target_names= df2['vaderSentimentLabel'].unique()))

plot_confusion_matrix(clf, X_test, Y_test)
plt.show()



# K = chi2_kernel(X_train, gamma=0.5)
# AK = additive_chi2_kernel(X_train)
# EU = euclidean_distances(X_train)

# Unable to process sparse matrices
# Only 2 Tolerances due to time needed for the experiment

for c in C:
    # Fit the model and predicct the output
    clf = LinearSVC(C=c).fit(X_train, Y_train)

    scv_pred = clf.predict(X_test)

    print("Linear SVC Accuracy: %s" % str(clf.score(X_test, Y_test)))
    print("Linear SVC F1-score: %s" % str(f1_score(Y_test, scv_pred, average='macro', zero_division=1)))
    print("Confusion Matrix")
    print(confusion_matrix(scv_pred, Y_test))

    print(classification_report(Y_test,scv_pred,target_names= df2['vaderSentimentLabel'].unique()))

    plot_confusion_matrix(clf, X_test, Y_test)
    plt.show()


# Fit the model and predict the output
clf = SVC(kernel='rbf', C=c, degree=2).fit(X_train, Y_train)

scv_pred = clf.predict(X_test)

print("SVC Accuracy: %s" % str(clf.score(X_test, Y_test)))
print("SVC F1-score: %s" % str(f1_score(Y_test, scv_pred, average='macro', zero_division=1)))
print("Confusion Matrix")
print(confusion_matrix(scv_pred, Y_test))

print(classification_report(Y_test,scv_pred,target_names= df2['vaderSentimentLabel'].unique()))

plot_confusion_matrix(clf, X_test, Y_test)
plt.show()


# Fit the Random Forest model and predict the output
clf1 = RandomForestClassifier(n_estimators=200).fit(X_train, Y_train)  # posa features ana split ?

rf_pred = clf1.predict(X_test)

print("Random Forest Accuracy: %s" % str(clf1.score(X_test, Y_test)))
print("SVC F1-score: %s" % str(f1_score(Y_test, rf_pred, average='macro', zero_division=1)))

print("Confusion Matrix")
conf_mat = confusion_matrix(rf_pred, Y_test)
print(conf_mat)

plot_confusion_matrix(clf1, X_test, Y_test)
plt.show()

print(classification_report(Y_test,rf_pred,target_names= df2['vaderSentimentLabel'].unique()))
