import csv
import numpy as np
from sklearn import svm,cross_validation
from sklearn.model_selection import train_test_split,cross_validate,cross_val_score


x_list=[]
y_list=[]
z_list=[]

x_file=open("x-tra.csv","r")
f=csv.reader(x_file)
for r in f:
	x_list.append(r)
x_file.close()


y_file=open("y-tra.csv","r")
F=csv.reader(y_file)
for r in F:
	y_list.extend(r)
y_file.close()


z_file=open("x-test.csv","r")
fi=csv.reader(z_file)
for r in fi:
	z_list.append(r)
z_file.close()

print(np.shape(z_list))
print("---")


x_train,x_test,y_train,y_test=train_test_split(x_list,y_list,test_size=0.2,shuffle=True)

model=svm.SVC()
model.fit(x_train,y_train)
score=cross_val_score(model,x_list,y_list,cv=10)
print(np.mean(score))

pre=model.predict(z_list)
print(len(pre))

#miss=0
#print(pre)
#print(y_list)
#for k in range(0,len(pre)):
#	if(pre[k]!=y_list[k]):
#		miss=miss+1
#print(miss,k+1,miss/(k+1))

print("------")
print(",".join(map(str, pre)))
#print(np.mean(score))
#print(slf)


#for i in range(0,240):
#	print(np.shape(x_train[i]))
	#clf = svm.SVC()
	#clf.fit(x_train[i], y_train[i])
	#print(clf.predict(x_test[i]))
	#print("correct:{}".format(y_test[i]))
