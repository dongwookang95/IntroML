
# %%
import mglearn
import sklearn
mglearn.plots.plot_scaling()
# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, 
cancer.target, random_state=1)
print(X_train.shape)
print(X_test.shape)

# %%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# %%
scaler.fit(X_train)
# %%
X_train_scaled = scaler.transform(X_train)

print("transformed shape: {}".format(X_train_scaled.shape))
print("pre-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
print("pre-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
print("pre-feature minimum after scaling:\n {}".format(X_train_scaled.min(axis=0)))
print("pre-feature maximum after scaling:\n {}".format(X_train_scaled.max(axis=0)))

#%%
#transform test data

X_test_scaled = scaler.transform(X_test)
# rpint test data properties after scaling
print("per-feature minimum after scaling:\n{}".format(X_test_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n{}".format(X_test_scaled.max(axis=0)))


# %%
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
# make synthetic data
X,_ = make_blobs(n_samples = 50, centers = 5, random_state=4, cluster_std=2)
# split it into training and test sets
X_train,X_test = train_test_split(X, random_state=5, test_size=.1)

#plot the training and test sets
fig, axes = plt.subplots(1,3,figsize=(13,4))

axes[0].scatter(X_train[:,0],X_train[:,1],c=mglearn.cm2(0),label="training set",s=60)
axes[0].scatter(X_test[:,0],X_test[:,1],marker='^',c=mglearn.cm2(1),label="test set",s=60)
axes[0].legend(loc='upper left')
axes[0].set_title("Original Data")

#scale the data using MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

axes[1].scatter(X_train_scaled[:,0],X_train_scaled[:,1],c=mglearn.cm2(0),label="training set", s=60)
axes[1].scatter(X_test_scaled[:,0],X_test_scaled[:,1],marker='^',c=mglearn.cm2(1),label="test set", s=60)
axes[1].set_title("scaled data")

#rescale the test set separately
#so test set min is 0 and test set max it 1
# Do not do this! for illustration purpose only

scaler = MinMaxScaler()
scaler.fit(X_test)

X_test_scaled_badly = scaler.transform(X_test)

axes[2].scatter(X_train_scaled[:,0],X_train_scaled[:,1],c=mglearn.cm2(0),label="training set", s=60)
axes[2].scatter(X_test_scaled_badly[:,0],X_test_scaled_badly[:,1],marker='^',c=mglearn.cm2(1),label="test set", s=60)
axes[2].set_title("Improperly scaled data")

for ax in axes:
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")

fig.tight_layout()

# %%
# This is Shortcuts and Efficient Alternatives for above procedure
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

#calling fit and transform in sequence (using method chaining)
X_scaled = scaler.fit(X_train).transform(X_train)
#same result, but more efficient computation
X_scaled_d = scaler.fit_transform(X_train)

# %%
from sklearn.svm import SVC
# %%
