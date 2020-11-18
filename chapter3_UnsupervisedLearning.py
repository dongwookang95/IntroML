
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

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,random_state=0)

svm=SVC(C=100)
svm.fit(X_train,y_train)
print("Test set accuracy: {:.2f}".format(svm.score(X_test,y_test)))

# %%
#preprocessing using 0-1 scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit(X_train).transform(X_train)
X_test_scaled = scaler.transform(X_test)

#learning an SVM on the scaled training data
svm.fit(X_train_scaled,y_train)

#scoring on the scaled test set
print("Scaled test set accuracy : {:.2f}".format(svm.score(X_test_scaled,y_test)))
# %%
#preprocessing using zero mean and unit variance scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train).transform(X_train)
X_test_scaled = scaler.transform(X_test)
#learning an SVM on the scaled training data
svm.fit(X_train_scaled, y_train)
print("SVM test accuracy: {:.2f}".format(svm.score(X_test_scaled, y_test)))




# %%
mglearn.plots.plot_pca_illustration()


# %%
import numpy as np

fig, axes= plt.subplots(15,2,figsize=(10,20))
malignant=cancer.data[cancer.target==0] 
benign = cancer.data[cancer.target==1]

ax= axes.ravel()

for i in range(30):
    _, bins = np.histogram(cancer.data[:,i], bins=50)
    ax[i].hist(malignant[:,i],bins=bins, color=mglearn.cm3(0),alpha=.5)
    ax[i].hist(benign[:,1],bins=bins, color=mglearn.cm3(2),alpha=.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["malignant","benign"],loc="best")
fig.tight_layout()

# %%
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

scaler = StandardScaler()
X_scaled = scaler.fit(cancer.data).transform(cancer.data)






# %%

# %%
from sklearn.decomposition import PCA
#keep the first two principal components of the data
pca = PCA(n_components=2)
#fit PCA model to breast cancer data
pca.fit(X_scaled)

#transform data onto the first two principal components
X_pca = pca.transform(X_scaled)
print("Original shape:{}".format(str(X_scaled.shape)))
print("Reduced shape: {}".format(str(X_pca.shape)))

# %%
#plot first vs. second principal component,colored by class
plt.figure(figsize=(8,8))
mglearn.discrete_scatter(X_pca[:,0],X_pca[:,1],cancer.target)
plt.legend(cancer.target_names,loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")

# %%
print("PCA component shape:{}".format(pca.components_.shape))

# %%
print("PCA component :{}".format(pca.components_))
# %%
plt.matshow(pca.components_,cmap='viridis')
plt.yticks([0,1],["First component","Second component"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
plt.xlabel("Feature")
plt.ylabel("Principal components")


# %%
from sklearn.datasets import fetch_lfw_people

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

fig, axes = plt.subplots(2,5,figsize=(15,8),subplot_kw={'xticks':(),'yticks':()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])



# %%
print("people.images.shape:{}".format(people.images.shape))
print("Number of classes:{}".format(len(people.target_names)))


#count how often each target appears
counts = np.bincount(people.target)
#print counts next to target names
for i, (count, name) in enumerate(zip(counts,people.target_names)):
    print("{0:25}{1:3}".format(name,count),end='  ')
    if (i+1)%3 == 0:
        print()

mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]]=1
X_people = people.data[mask]
y_people = people.target[mask]
# scale the grayscale values to be between 0 and 1
# instead of 0 and 255 for better numeric stability
X_people=X_people/255



# %%
from sklearn.neighbors import KNeighborsClassifier
#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_people,y_people,stratify=y_people,random_state=0)
# build a KNeighborsClassifier using one neighbor
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("Test set score of 1-nn:{:.2f}".format(knn.score(X_test,y_test)))


# %%
mglearn.plots.plot_pca_whitening()
# %%
pca=PCA(n_components=100, whiten=True,random_state=0)
X_train_pca=pca.fit(X_train).transform(X_train)
X_test_pca = pca.transform(X_test)

print("X_train_pca.shape:{}".format(X_train_pca.shape))



# %%
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca,y_train)
print("Test set accuracy:{:.2f}".format(knn.score(X_test_pca,y_test)))


# %%
print("pca.components_.shape:{}".format(pca.components_.shape))


# %%
fix, axes = plt.subplots(3,5,figsize=(15,12),subplot_kw={'xticks':(), 'yticks':()})
for i, (component, ax) in enumerate(zip(pca.components_,axes.ravel())):
    ax.imshow(component.reshape(image_shape), cmap='viridis')
    ax.set_title("{}.component".format((i+1)))

# %%
mglearn.plots.plot_pca_faces(X_train,X_test,image_shape)
# %%
mglearn.discrete_scatter(X_train_pca[:,0], X_train_pca[:,1],y_train)
plt.xlabel("First principal component")
plt.ylabel("Second principal component")

# %%
mglearn.plots.plot_nmf_illustration()


# %%
mglearn.plots.plot_nmf_faces(X_train, X_test,image_shape)

# %%
from sklearn.decomposition import NMF
nmf = NMF(n_components=15, random_state=0)
nmf.fit(X_train)
X_train_nmf=nmf.transform(X_train)
X_test_nmf=nmf.transform(X_test)

fix, axes = plt.subplots(3,5,figsize=(15,12), subplot_kw={'xticks':(),'yticks':()})

for i, (component, ax) in enumerate(zip(nmf.components_,axes.ravel())):
    ax.imshow(component.reshape(image_shape))
    ax.set_title("{}.component".format(i))
# %%
compn = 3
# sort by 3rd component, plot first 10 images
inds = np.argsort(X_train_nmf[:,compn])[::-1]
fig,axes = plt.subplots(2,5,figsize=(15,8), subplot_kw={'xticks':(),'yticks':()})

fig.suptitle("Large component 3")
for i,(ind ,ax) in enumerate(zip(inds,axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape))

compn = 7
# sort by 7th component, plot first 10 images
inds = np.argsort(X_train_nmf[:,compn])[::-1]
fig,axes = plt.subplots(2,5,figsize=(15,8), subplot_kw={'xticks':(),'yticks':()})

fig.suptitle("Large component 7")
for i,(ind ,ax) in enumerate(zip(inds,axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape))
# %%

S = mglearn.datasets.make_signals()
plt.figure(figsize=(6,1))
plt.plot(S, '-')
plt.xlabel("Time")
plt.ylabel("Signal")
# %%
#mix data into a 100 dimensional state
A = np.random.RandomState(0).uniform(size=(100,3))
X = np.dot(S,A.T)
print("Shape of measurements: {}".format(X.shape))
# %%
nmf = NMF(n_components=3, random_state=42)
S_=nmf.fit_transform(X)
print("Recoverd signal shape: {}".format(S_.shape))
# %%
pca = PCA(n_components=3)
H = pca.fit_transform(X)



# %%
models = [X,S,S_,H]
names = ['Observations (first three measurements)', 'True sources', 'NMF recovered signals','PCA recovered signals']

fig, axes = plt.subplots(4,figsize=(8,4),gridspec_kw={'hspace':.5},subplot_kw={'xticks':(),'yticks':()})

for model, name, ax in zip(models, names, axes):
    ax.set_title(name)
    ax.plot(model[:,:3],'-')

    
# %%

from sklearn.datasets import load_digits
digits = load_digits()

fig, axes = plt.subplots(2,5,figsize=(10,5), subplot_kw={'xticks':(),'yticks':()})
for ax, img in zip(axes.ravel(),digits.images):
    ax.imshow(img)

# %%

pca = PCA(n_components=2)
