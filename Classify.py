from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC,LinearSVC


#Classify users gender based on their height, weight and shoe size
#[h,w,s]
X = [[180,100,44],[99,12,25],[89,66,20],[100,22,40],[178,60,22],[190,33,42],[160,60,44],[94,12,25],[86,66,20],[89,22,40],[168,60,22],[180,33,42]]
Y = ['male','female','female','male','male','male','female','male','female','male','female','female']
P= [[150,34,44],[180,100,43],[144,62,35],[144,64,33],[188,88,44]]
Q = ['female','male','male','female','male']

classify = tree.DecisionTreeClassifier()
classify = classify.fit(X,Y)
predict_dtc = classify.predict(P)
print("Result from using Decision Tree Classifier:"+ str(predict_dtc))


classify = tree.ExtraTreeClassifier()
classify = classify.fit(X,Y)
predict_etx = classify.predict(P)

print('Using Extra Tree Classifier:'+str(predict_etx))

classify = GaussianNB()
classify = classify.fit(X,Y)
predict_NB= classify.predict(P)
print("Result from using Naive Bayes:"+ str(predict_NB))


classify = SVC()
classify = classify.fit(X,Y)
predict_svc = classify.predict(P)

print('Using Support Vector Classifier:'+str(predict_svc))

classify = LinearSVC()
classify = classify.fit(X,Y)
predict_linear_svc = classify.predict(P)

print('Using Support Vector Classifier:'+str(predict_linear_svc))

#analyze accuracy of both classifiers
print ('accuracy of DTC,', accuracy_score(Q,predict_dtc))
print ('accuracy of ExTC,' ,accuracy_score(Q,predict_etx))
print ('accuracy of NB,', accuracy_score(Q,predict_NB))
print ('accuracy of SVC,' ,accuracy_score(Q,predict_svc))
print ('accuracy of LinearSVC,', accuracy_score(Q,predict_linear_svc))




