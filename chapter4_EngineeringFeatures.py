#%%
import mglearn
import pandas as pd
import os
# the file has no headers naming the columns, so we pass header=none
# and provide the column names explicitly in "names"

adult_path = os.path.join(mglearn.datasets.DATA_PATH,"adult.data")
data = pd.read_csv(adult_path,header=None, index_col=False,
names=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race',
'gender','capital-gain','capital-loss','hours-per-week','native-country','income'])
# For illustration purposes, we only select some of the columes
data = data[['age','workclass','education','gender','hours-per-week','occupation','income']]
# IPython.display allows nice output formatting within the Jupyter notebook
display(data.head())

# %%
print(data.gender.value_counts())
# %%
print("Original features:\n", list(data.columns),"\n")
data_dummies = pd.get_dummies(data)
print("Featurs after get_dummies:\n",list(data_dummies.columns))
# %%
data_dummies.head()
# %%
features = data_dummies.loc[:,'age':'occupation_ Transport-moving']
# Extract NumPy arrays
X = features.values
y = data_dummies['income_ >50K'].values
print("X.shape:{} y.shape:{}".format(X.shape, y.shape))


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X,y,random_state=0)
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
print("Test score:{:.2f}".format(logreg.score(X_test,y_test)))
# %%
#create a DataFrame with an integer feature and a categorical string feature
demo_df = pd.DataFrame({'Integer Feature':[0,1,2,1],'Categorical Feature':['socks','fox','socks','box']})
display(demo_df)

# %%
display(pd.get_dummies(demo_df))
# %%
demo_df['Integer Feature'] = demo_df['Integer Feature'].astype(str)
display(pd.get_dummies(demo_df,columns=['Integer Feature','Categorical Feature']))


# %%
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
import numpy as np

X,y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3,3,1000,endpoint=False).reshape(-1,1)

reg = DecisionTreeRegressor(min_samples_split=3).fit(X,y)
plt.plot(line,reg.predict(line),label="decision tree")



reg = LinearRegression().fit(X,y)
plt.plot(line,reg.predict(line),label="Linear regression")

plt.plot(X[:,0],y,'o',c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")
# %%

bins = np.linspace(-3,3,11)
print("bins:{}".format(bins))

# %%
which_bin = np.digitize(X,bins=bins)
print("\nData points:\n",X[:5])
print("\nBin membership for data points:\n", which_bin[:5])
# %%
from sklearn.preprocessing import OneHotEncoder
# transform using the OneHotEncoder
encoder = OneHotEncoder(sparse=False)
# encoder.fit finds the unique values that appear in which_bin
encoder.fit(which_bin)
#transform creates the one-hot encoding
X_binned = encoder.transform(which_bin)
print(X_binned[:5])
# %%
print("X_binned.shape:{}".format(X_binned.shape))
# %%
line_binned = encoder.transform(np.digitize(line,bins=bins))
reg=LinearRegression().fit(X_binned,y)
plt.plot(line,reg.predict(line_binned),label='linear regression binned')

reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned,y)
plt.plot(line,reg.predict(line_binned),label='decision tree binned')

plt.plot(X[:,0],y,'o',c='k')
plt.vlines(bins,-3,3,linewidth=1,alpha=.2)
plt.legend(loc="best")
plt.ylabel("Regression output")
plt.xlabel("Input feature")
# %%
X_combined = np.hstack([X,X_binned])
print(X_combined.shape)

# %%
reg = LinearRegression().fit(X_combined,y)

line_combined = np.hstack([line, line_binned])
plt.plot(line,reg.predict(line_combined),label='linear regression combined')

for bin in bins:
    plt.plot([bin,bin],[-3,3],':',c='k',linewidith=1)

plt.legend(loc="best")
plt.ylabel("Regrssion output")
plt.xlabel("Input feature")
plt.plot(X[:,0],y,'o',c='k')
# %%
X_product = np.hstack([X_binned, X*X_binned])
print(X_product.shape)

# %%
reg = LinearRegression().fit(X_product,y)

line_product = np.hstack([line_binned,line*line_binned])
plt.plot(line, reg.predict(line_product),label='linear regression product')

for bin in bins:
    plt.plot([bin,bin],[-3,3],':',c='k',linewidth=1)

plt.plot(X[:,0],y,'o',c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")
# %%
from sklearn.preprocessing import PolynomialFeatures

# include polynomials up to x ** 10:
# the default "include_bias = true" adds a feature that's constantly 1
poly = PolynomialFeatures(degree=10, include_bias=False)
poly.fit(X)
X_poly = poly.transform(X)
# %%
print("X_poly.shape:{}".format(X_poly.shape))
# %%
print("Entries of X:\n{}".format(X[:5]))
np.set_printoptions(suppress=True)
print("Entries of X_ploy:\n{}".format(X_poly[:5]))
# %%
print("Polynomial feature names:\n{}".format(poly.get_feature_names()))
# %%
reg = LinearRegression().fit(X_poly,y)

line_poly = poly.transform(line)
plt.plot(line, reg.predict(line_poly),label='polynomial linear regression')
plt.plot(X[:,0],y,'o',c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")

# %%
from sklearn.svm import SVR

for gamma in [1,10]:
    svr = SVR(gamma=gamma).fot(X,y)
    plt.plot(line,svr.predict(line),label='SVR gamma={}'.format(gamma))
plt.plt(X[:,0],y,'o',c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")

