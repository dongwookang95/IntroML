#%%
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# load and split the data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
# compute minimum and maximum on the training data
scaler = MinMaxScaler().fit(X_train)
# %%
# rescale the training data
X_train_scaled = scaler.transform(X_train)

svm = SVC()
#learn an SVM on the scaled training data
svm.fit(X_train_scaled,y_train)
#scale the test data and score the scaled data
X_test_scaled = scaler.transform(X_test)
print("Test score: {:.2f}".format(svm.score(X_test_scaled,y_test)))
# %%
import mglearn

mglearn.plots.plot_improper_processing()

# %%
from sklearn.pipeline import Pipeline
pipe = Pipeline([("scaler",MinMaxScaler()),("svm",SVC())])
# %%
pipe.fit(X_train,y_train)
# %%
print("Test score: {:.2f}".format(pipe.score(X_test,y_test)))
# %%
param_grid = {'svm__C': [0.001,0.01,0.1,1,10,100],'svm__gamma': [0.001,0.01,0.1,1,10,100]}
# %%
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipe, param_grid=param_grid,cv=5)
grid.fit(X_train,y_train)
print("Best cross-validation accuracy:{:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test,y_test)))
print("Best parameters : {}".format(grid.best_params_))
# %%
mglearn.plots.plot_proper_processing()
# %%
def fit(self,X,y):
    X_transformed = X
    for name,estimator in self.steps[:-1]:
        #iterate over all but the final step
        #fit and transform the data
        X_transformed = estimator.fit_transform(X_transformed,y)
    #fit the last step
    self.steps[-1][1].fit(X_transformed,y)
    return self
# %%
def predict(self,X):
    X_transformed = X
    for step in self.steps[:-1]:
        #iterate over all but the final step
        #transform the data
        X_transformed = step[1].transform(X_transformed)
    # predict using the last step
    return self.steps[-1][1].predict(X_transformed)
# %%
from sklearn.pipeline import make_pipeline
#standard syntax
pipe_long = Pipeline([("scaler",MinMaxScaler()),("svm",SVC(C=100))])
#abbreviated syntax
pipe_short = make_pipeline(MinMaxScaler(),SVC(C=100))
# %%
print("Pipeline steps:\n{}".format(pipe_short.steps))
# %%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pipe = make_pipeline(StandardScaler(),PCA(n_components=2),StandardScaler())
print("Pipeline steps:\n{}".format(pipe.steps))
# %%


#fit the pipeline defined before to the cancer dataset
pipe.fit(cancer.data)
#extract the first two principal components from the "pca" step
components = pipe.named_steps["pca"].components_
print("components.shape:{}".format(components.shape))
# %%
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(StandardScaler(),LogisticRegression())

# %%
param_grid = {'logisticregression__C':[0.01,0.1,1,10,100]}
# %%
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,random_state=4)
grid=GridSearchCV(pipe,param_grid,cv=5)
grid.fit(X_train,y_train)
# %%
print("Best estimator:\n{}".format(grid.best_estimator_))
# %%
print("Logistic regression step:\n{}".format(grid.best_estimator_.named_steps["logisticregression"]))
# %%
print("Logistic regression coefficients:\n{}".format(grid.best_estimator_.named_steps["logisticregression"].coef_))


# %%
from sklearn.datasets import load_boston
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target,
                                                    random_state=0)

from sklearn.preprocessing import PolynomialFeatures
pipe = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(),
    Ridge())
# %%
param_grid = {'polynomialfeatures__degree': [1, 2, 3],
              'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# %%
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
# %%

# %%
from matplotlib import pyplot as plt

plt.matshow(grid.cv_results_['mean_test_score'].reshape(3,-1),vmin=0,cmap='viridis')
plt.xlabel("ridge__alpha")
plt.ylabel("polynomialfeatures__degree")

plt.xticks(range(len(param_grid['ridge__alpha'])), param_grid['ridge__alpha']) 
plt.yticks(range(len(param_grid['polynomialfeatures__degree'])),param_grid['polynomialfeatures__degree'])
plt.colorbar()
# %%
print("Best parameters : {}".format(grid.best_params_))


# %%
print("Test-set score:{:.2f}".format(grid.score(X_test,y_test)))
# %%
param_grid = {'ridge__alpha':[0.001,0.01,0.1,1,10,100]}
pipe = make_pipeline(StandardScaler(),Ridge())
grid=GridSearchCV(pipe,param_grid,cv=5)
grid.fit(X_train,y_train)
print("Score without poly features:{:.2f}".format(grid.score(X_test,y_test)))
# %%
pipe = Pipeline([('preprocessing',StandardScaler()),('classifier',SVC())])
# %%
from sklearn.ensemble import RandomForestClassifier

param_grid = [
    {'classifier': [SVC()], 'preprocessing': [StandardScaler(), None],
     'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
     'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
    {'classifier': [RandomForestClassifier(n_estimators=100)],
     'preprocessing': [None], 'classifier__max_features': [1, 2, 3]}]
# %%
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best params:\n{}\n".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))
# %%
