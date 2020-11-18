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

# %%

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
    svr = SVR(gamma=gamma).fit(X,y)
    plt.plot(line,svr.predict(line),label='SVR gamma={}'.format(gamma))
plt.plot(X[:,0],y,'o',c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")




# %%
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

#rescale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# %%
poly = PolynomialFeatures(degree=2).fit(X_train_scaled)
X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
print("X_train.shape:{}".format(X_train.shape))
print("X_train_poly.shape:{}".format(X_train_poly.shape))
# %%
print("Polynomial feature names:\n{}".format(poly.get_feature_names()))


# %%
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train_scaled,y_train)
print("Score without interactions:{:.3f}".format(ridge.score(X_test_scaled, y_test)))
ridge = Ridge().fit(X_train_poly,y_train)
print("Score with interactions : {:.3f}".format(ridge.score(X_test_poly,y_test)))


# %%
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100).fit(X_train_scaled,y_train)
print("Score without interactions:{:.3f}".format(rf.score(X_test_scaled,y_test)))
rf = RandomForestRegressor(n_estimators=100).fit(X_train_poly, y_train)
print("Score without interactions:{:.3f}".format(rf.score(X_test_poly,y_test)))
# %%
rnd = np.random.RandomState(0)
X_org=rnd.normal(size=(1000,3))
w = rnd.normal(size=3)

X=rnd.poisson(10*np.exp(X_org))
y=np.dot(X_org,w)

# %%
print("Number if feature appearances:\n{}".format(np.bincount(X[:,0])))

# %%
bins= np.bincount(X[:,0])
plt.bar(range(len(bins)),bins,color='b')
plt.ylabel("Number of appearances")
plt.xlabel("Value")
# %%
from sklearn.linear_model import Ridge

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
score = Ridge().fit(X_train,y_train).score(X_test,y_test)
print("Test score:{:.3f}".format(score))

# %%
X_train_log = np.log(X_train+1)
X_test_log = np.log(X_test+1)
# %%
plt.hist(X_train_log[:,0],bins=25, color='gray')
plt.ylabel("Number of appearances")
plt.xlabel("Value")
# %%
score = Ridge().fit(X_train_log,y_train).score(X_test_log,y_test)
print("Test score:{:.3f}".format(score))

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
# get deterministic random numbers
rng = np.random.RandomState(42)
noise  = rng.normal(size=(len(cancer.data),50))
# add noise features to the data
# the first 30 features are from the dataset, the next 50 are noise
X_w_noise = np.hstack([cancer.data,noise])

X_train, X_test,y_train,y_test = train_test_split(X_w_noise, cancer.target, random_state=0,test_size=.5)
#use f_classif(the default) and SelectPercentile to select 50% of features
select = SelectPercentile(percentile=50)
select.fit(X_train,y_train)
#transform training set
X_train_selected = select.transform(X_train)

print("X_train.shape:{}".format(X_train.shape))
print("X_train_selected.shape:{}.".format(X_train_selected.shape))
# %%
mask=select.get_support()
print(mask)
#visualize the mask - - black is True, white is False
plt.matshow(mask.reshape(1,-1),cmap='gray_r')
plt.xlabel("Sample index")
plt.yticks(()
)


# %%
from sklearn.linear_model import LogisticRegression

#transform test data

X_test_selected = select.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train,y_train)
print("Score with all features: {:.3f}".format(lr.score(X_test, y_test)))
lr.fit(X_train_selected, y_train)
print("Score with only selected features: {:.3f}".format(
lr.score(X_test_selected, y_test)))

# %%
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median")
# %%
select.fit(X_train,y_train)
X_train_l1 = select.transform(X_train)
print("X_train.shape:{}".format(X_train.shape))
print("X_train_l1.shape:{}".format(X_train_l1.shape))
# %%
mask = select.get_support()
#visualize the mask  - - black is True, white is False
plt.matshow(mask.reshape(1,-1),cmap='gray_r')
plt.xlabel("Sample index")
plt.yticks(())
# %%
X_test_l1 = select.transform(X_test)
score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1,y_test)
print("Test score:{:.3f}".format(score))
# %%
from sklearn.feature_selection import RFE
select = RFE(RandomForestClassifier(n_estimators=100,random_state=42),n_features_to_select=40)
select.fit(X_train,y_train)
#visualise the selected features:
mask = select.get_support()

plt.matshow(mask.reshape(1,-1),cmap='gray_r')
plt.xlabel("Sample index")
plt.yticks(())
# %%
print("Test scoreL {:.3f}".format(select.score(X_test,y_test)))
# %%
citibike = mglearn.datasets.load_citibike()
# %%
print("Citi Bike data:\n{}".format(citibike.head()))
# %%
plt.figure(figsize=(10,3))
xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(),freq="D")
plt.xticks(xticks,xticks.strftime("%a %m-%d"),rotation=90,ha="left")
plt.plot(citibike,linewidth=1)
plt.xlabel("Date")
plt.ylabel("Rentals")


# %%
#extract the target values (number oif rentals)

y = citibike.values

#convert to POSIX time by dividing by 10**9
X = citibike.index.astype("int64").values.reshape(-1,1)//10**9




# %%
# use the first 184 data points for training, and the rest for testing
n_train = 184
#function to evaluate and plot a regressor on a given feature set
def eval_on_features(features, target, regressor):
    #split the given featurs into a training and a test set
    X_train,X_test = features[:n_train],features[n_train:]
    # also split the target array
    y_train,y_test = target[:n_train],target[n_train:]
    regressor.fit(X_train,y_train)
    print("Test0set R^2:{:.2f}".format(regressor.score(X_test,y_test)))
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    plt.figure(figsize=(10,3))
    plt.xticks(range(0,len(X),8),xticks.strftime("%a %m -%d"), rotation=90,ha="left")

    plt.plot(range(n_train),y_train,label="train")
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label="test")
    plt.plot(range(n_train), y_pred_train, '--', label="prediction train")
    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--', label="prediction test")
    plt.legend(loc=(1.01, 0))
    plt.xlabel("Date")
    plt.ylabel("Rentals")

# %%
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100,random_state=0)
eval_on_features(X,y,regressor)
# %%
X_hour = citibike.index.hour.values.reshape(-1,1)
eval_on_features(X_hour,y,regressor)
# %%
X_hour_week = np.hstack([citibike.index.dayofweek.values.reshape(-1,1),citibike.index.hour.values.reshape(-1,1)])
eval_on_features(X_hour_week,y,regressor)
# %%
from sklearn.linear_model import LinearRegression
eval_on_features(X_hour_week,y,LinearRegression())
# %%
enc = OneHotEncoder()
X_hour_week_onehot=enc.fit_transform(X_hour_week).toarray()
# %%
eval_on_features(X_hour_week_onehot, y, Ridge())

# %%
poly_transformer = PolynomialFeatures(degree=2,interaction_only=True, include_bias=False)
X_hour_week_onehot_poly=poly_transformer.fit_transform(X_hour_week_onehot)
lr = Ridge()
eval_on_features(X_hour_week_onehot_poly,y,lr)
# %%
hour = ["%02d:00" % i for i in range(0, 24, 3)]
day = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
features = day + hour

# %%
features_poly = poly_transformer.get_feature_names(features) 
features_nonzero = np.array(features_poly)[lr.coef_ != 0] 
coef_nonzero = lr.coef_[lr.coef_ != 0]
# %%
plt.figure(figsize=(15, 2))
plt.plot(coef_nonzero, 'o')
plt.xticks(np.arange(len(coef_nonzero)), features_nonzero, rotation=90)
plt.xlabel("Feature name")
plt.ylabel("Feature magnitude")
# %%
