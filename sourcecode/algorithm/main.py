import numpy as np
feature=np.loadtxt("feature.csv",delimiter=',')
id=np.loadtxt("id.csv",delimiter=',')
from sklearn.naive_bayes import GaussianNB  
clf = GaussianNB().fit(feature, id)
eval=np.loadtxt("eval.csv",delimiter=',')
result = clf.predict([eval])
print result