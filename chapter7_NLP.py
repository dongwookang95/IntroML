#%%
from sklearn.datasets import load_files

reviews_train = load_files("aclImdb/train")
text_train,y_train = reviews_train.data,reviews_train.target

print("type of text_train:{}".format(type(text_train)))
print("length of text_train:{}".format(len(text_train)))
print("test_train[6]:\n{}".format(text_train[6]))
# %%
text_train = [doc.replace(b"<br /",b" ")for doc in text_train]

# %%
import numpy as np
print("Samples per class(training):{}".format(np.bincount(y_train)))
# %%
reviews_test = load_files("aclImdb/test")

text_test,y_test = reviews_test.data,reviews_test.target
print("Number of documents in test data:{}".format(len(text_test)))
print("Samples per class(test):{}".format(np.bincount(y_test)))
text_test = [doc.replace(b"<br /",b" ")for doc in text_test]
# %%
bards_words = ["THe fool doth think he is wise,","but the wise man knows himself to be a fool"]
# %%
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(bards_words)
# %%
print("Vocabulary size:{}".format(len(vect.vocabulary_)))
print("Vocabulary content:\n {}".format(vect.vocabulary_))
# %%
bag_of_words = vect.transform(bards_words)
print("bag_of_words:{}".format(repr(bag_of_words)))
# %%
print("Dense representation of bag_of_words:\n{}".format(bag_of_words.toarray()))
# %%
vect = CountVectorizer().fit(text_train)
X_train = vect.transform(text_train)
print("X_train:\n{}".format(repr(X_train)))
# %%
feature_names = vect.get_feature_names()
print("Number of features:{}".format(len(feature_names)))
print("First 20 features:\n{}".format(feature_names[:20]))
print("Features 20010 to 20030:\n{}".format(feature_names[20010:20030]))
print("Every 2000th features:\n{}".format(feature_names[::2000]))
# %%
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
scores = cross_val_score(LogisticRegression(),X_train,y_train,cv=5)
print("Mean cross-validation accuracy:{:.2f}".format(np.mean(scores)))
# %%
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.001,0.01,0.1,1,10]}
grid = GridSearchCV(LogisticRegression(),param_grid,cv=5)
grid.fit(X_train,y_train)
print("Best cross-validation score:{:.2f}".format(grid.best_score_))
print("Best parameters: ",grid.best_params_)
# %%
X_test = vect.fit_transform(text_test)
print("Test score:{:.2f}".format(grid.score(X_test,y_test)))
# %%
vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)
print("X_train with min_df:{}".format(repr(X_train)))
# %%
feature_names = vect.get_feature_names()

print("First 20 features:\n{}".format(feature_names[:50]))
print("Features 20010 to 20030:\n{}".format(feature_names[20010:20030]))
print("Every 2000th features:\n{}".format(feature_names[::700]))
# %%
grid = GridSearchCV(LogisticRegression(),param_grid,cv=5)
grid.fit(X_train,y_train)
print("Best cross-validation score:{:.2f}".format(grid.best_score_))
# %%

# %%
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
print("Number of stop words:{}".format(len(ENGLISH_STOP_WORDS)))
print("Every 10th stop words:\n{}".format(list(ENGLISH_STOP_WORDS)[::10]))
# %%
# Specifying stop_words="english"uses the built-in list.
# We could also augment it and pass our own.
vect = CountVectorizer(min_df=5,stop_words="english").fit(text_train)
X_train = vect.transform(text_train)
print("X_train with stop words:\n{}".format(repr(X_train)))
# %%
grid = GridSearchCV(LogisticRegression(),param_grid, cv=5)
grid.fit(X_train,y_train)
print("Best cross-validation scroe:{:.2f}".format(grid.best_score_))
# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
param_grid={'logisticregression__C':[0.001,0.1,0.1,1,10]}
grid = GridSearchCV(pipe,param_grid,cv=5)
grid.fit(text_train,y_train)
print("Best cross-validation score:{:.2f}".format(grid.best_score_))
# %%
vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]
# transform the traning dataset
X_train = vectorizer.transform(text_train)
#find maximum value for each of the features over the dataset
max_value = X_train.max(axis=0).toarray().ravel()
sorted_by_tfidf = max_value.argsort()
#get feature names
feature_names = np.array(vectorizer.get_feature_names())

print("Features with lowest tfidf:\n{}".format(feature_names[sorted_by_tfidf[:20]]))
print("Features with highest tfidf:\n{}".format(feature_names[sorted_by_tfidf[-20:]]))
# %%
sorted_by_idf=np.argsort(vectorizer.idf_)
print("Features with lowest2 tfidf:\n{}".format(feature_names[sorted_by_tfidf[:100]]))



# %%
import mglearn
mglearn.tools.visualize_coefficients(grid.best_estimator_.named_steps["logisticregression"].coef_,feature_names,n_top_features=40)

# %%
print("bards_words:\n{}".format(bards_words))
# %%
cv = CountVectorizer(ngram_range=(1,1)).fit(bards_words)
print("Vocabulary size: {}".format(len(cv.vocabulary_)))
print("Vocabulary:\n{}".format(cv.get_feature_names()))

# %%
cv = CountVectorizer(ngram_range=(2,2)).fit(bards_words)
print("Vocabulary size:{}".format(len(cv.vocabulary_)))
print("Vocabulary:\n{}".format(cv.get_feature_names()))
# %%
print("Transformed data (dense):\n{}".format(cv.transform(bards_words).toarray()))
# %%
cv = CountVectorizer(ngram_range=(1,3)).fit(bards_words)
print("Vocabulary size:{}".format(len(cv.vocabulary_)))
print("Vocabulary:\n{}".format(cv.get_feature_names()))
# %%
pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
# running the grid-search takes a long time because of the
# relatively large grid and the inclusion of trigrams
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100],
              "tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)]}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters:\n{}".format(grid.best_params_))
# %%
from matplotlib import pyplot as plt

# extract scores from grid_search
scores = grid.cv_results_['mean_test_score'].reshape(-1, 3).T
# visualize heat map
heatmap = mglearn.tools.heatmap(
    scores, xlabel="C", ylabel="ngram_range", cmap="viridis", fmt="%.3f",
    xticklabels=param_grid['logisticregression__C'],
    yticklabels=param_grid['tfidfvectorizer__ngram_range'])
plt.colorbar(heatmap)
# %%
# extract feature names and coefficients
vect = grid.best_estimator_.named_steps['tfidfvectorizer']
feature_names = np.array(vect.get_feature_names())
coef = grid.best_estimator_.named_steps['logisticregression'].coef_
mglearn.tools.visualize_coefficients(coef, feature_names, n_top_features=40)
plt.ylim(-22, 22)
# %%
# find 3-gram features
mask = np.array([len(feature.split(" ")) for feature in feature_names])==3
# visualiza only 3-gram features
mglearn.tools.visualize_coefficients(coef.ravel()[mask],feature_names[mask],n_top_features=40)

#visualize only 3-gram
#%%
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz

# %%
import spacy
import nltk
import en_core_web_sm
nlp = en_core_web_sm.load()

# load spacy's English-language models
# instantiate nltk's Porter stemmer
stemmer = nltk.stem.PorterStemmer()

# define function to compare lemmatization in spacy with stemming in nltk
def compare_normalization(doc):
    # tokenize document in spacy
    doc_spacy = nlp(doc)
    # print lemmas found by spacy
    print("Lemmatization:")
    print([token.lemma_ for token in doc_spacy])
    # print tokens found by Porter stemmer)
    print("Stemming:")
    print([stemmer.stem(token.norm_.lower()) for token in doc_spacy])
# %%
compare_normalization(u"Our meeting today was worse than yesterday,""I'm scared of meeting the cliuents tomorrow.")
# %%
# Technicality : we want to use the regexp-based tokenizer
# that is used by CountVectorizer and only use the lemmatization
# from spacy. To this end, we replace en_nlp.tokenizer(the spacy tokenizer)
# with the regexp-based tokenization
# Technicallity: we want to use the regexp based tokenizer
# that is used by CountVectorizer  and only use the lemmatization
# from SpaCy. To this end, we replace en_nlp.tokenizer (the SpaCy tokenizer)
# with the regexp based tokenization
import re
# regexp used in CountVectorizer:
regexp = re.compile('(?u)\\b\\w\\w+\\b')
# load spacy language model
nlp = en_core_web_sm.load()
old_tokenizer = nlp.tokenizer
# replace the tokenizer with the preceding regexp
nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(
    regexp.findall(string))

# create a custom tokenizer using the SpaCy document processing pipeline
# (now using our own tokenizer)
def custom_tokenizer(document):
    doc_spacy = nlp(document)
    return [token.lemma_ for token in doc_spacy]

# define a count vectorizer with the custom tokenizer
lemma_vect = CountVectorizer(tokenizer=custom_tokenizer, min_df=5)
# %%
# transform text_train using CountVectorizer with lemmatization
X_train_lemma = lemma_vect.fit_transform(text_train)
print({"X_train_lemma.shape:{}".format(X_train_lemma.shape)})

# standard CountVectorizer for reference
vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)
print("X_train.shape:{}".format(X_train.shape))
# %%
# build a grid-search using only 1% of the data as training set:
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.99,
                            train_size=0.01, random_state=0)
grid = GridSearchCV(LogisticRegression(), param_grid, cv=cv)
# perform grid search with standard CountVectorizer
grid.fit(X_train, y_train)
print("Best cross-validation score "
      "(standard CountVectorizer): {:.3f}".format(grid.best_score_))
# perform grid search with Lemmatization
grid.fit(X_train_lemma, y_train)
print("Best cross-validation score "
      "(lemmatization): {:.3f}".format(grid.best_score_))
# %%
vect = CountVectorizer(max_features=10000,max_df=.15)
X = vect.fit_transform(text_train)
# %%
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=10, learning_method="batch",max_iter=25, random_state=0)

# We build the model and transform the data in one step
# Computing transform takes some time,
# and we can save time by doing both at once

document_topics = lda.fit_transform(X)
# %%

# %%

# %%

# %%

# %%
