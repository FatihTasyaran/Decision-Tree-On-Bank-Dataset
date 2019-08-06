from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn import metrics
import graphviz
##
import matplotlib.pyplot as plt


#f = open("/home/fatih/Documents/CS412/hw2/bank_additional/bank-additional/bank-additional-full.csv", "r")

##NOTES:
##-> Duration highly effects the outcome of prediction.
##   if duration=0 then y=no
##   therefore could be discarded

names = ["age", "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y"]

df2 = pd.read_csv("bank-additional-full.csv", sep=';')




##TAILORING##
df = df2.set_axis(names, axis=1, inplace=False)
df = df[df.job != "unknown"]
df = df[df.default != "unknown"]
df = df[df.poutcome != "nonexistent"]

del df['duration']

le = preprocessing.LabelEncoder()

#print(df.loc[41182])
#print(df.loc[41159])
#print(df.loc[24013])

for column in df:
    df[column] = le.fit_transform(df[column])

names2 = ["age", "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "campaign", "pdays", "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
##TAILORING##

#print (df)
#print(df.loc[41182])
#print(df.loc[41159])
#print(df.loc[24013])

X = np.array(df.drop(['y'], 1))
y = np.array(df['y'])

train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.2)

depth_accuracies = []
#depth_accuracies.append(0.750996015936255)

for i in range(1, 51):

    if (i < 50):
        clf = tree.DecisionTreeClassifier(max_depth=i)#max_depth=3)
    else:
        clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_X,train_y)

    if(i==5):
        clf_5 = tree.DecisionTreeClassifier(max_depth=i)#max_depth=3)
        clf_5 = clf_5.fit(train_X,train_y)

        '''
    if(i < 10):
    
        vis_tree = tree.export_graphviz(clf, out_file=None, feature_names=names2,
                                    filled=True, rounded=True)
        graph = graphviz.Source(vis_tree)
        graph.render("tree_"+str(i))
    
        '''
    predictions = (clf.predict(test_X))

    accuracy = metrics.accuracy_score(predictions,test_y)
    depth_accuracies.append(accuracy)
    
    print("Accuracy: ", accuracy)
    print("Depth with ", i, " done!")


'''
axis = []
for i in range(1,51):
    axis.append(i)

plt.plot(axis, depth_accuracies)
plt.ylabel("Accuracy")
plt.xlabel("Tree Depth")
plt.savefig("plotting.pdf")
plt.show()
'''

#################################################################
######################NOW WITH DURATIONS######################
#################################################################

##TAILORING##
df_dr = df2.set_axis(names, axis=1, inplace=False)

df_dr = df_dr[df_dr.job != "unknown"]
df_dr = df_dr[df_dr.default != "unknown"]
df_dr = df_dr[df_dr.poutcome != "nonexistent"]

#del df['duration']

le = preprocessing.LabelEncoder()

#print(df.loc[41182])
#print(df.loc[41159])
#print(df.loc[24013])

for column in df:
    df_dr[column] = le.fit_transform(df_dr[column])

names3_1 = ["age", "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]

X_DR = np.array(df_dr.drop(['y'], 1))
y_dr = np.array(df_dr['y'])

train_X_DR, test_X_DR, train_y_dr, test_y_dr = train_test_split(X_DR,y_dr,test_size=0.2)

depth_accuracies_dr = []
#depth_accuracies_dr.append(0.7141434262948207)

#print(df_dr)

for i in range(1, 51):

    if(i < 50):
        clf_dr = tree.DecisionTreeClassifier(max_depth=i)#max_depth=3)
    else:
        clf_dr = tree.DecisionTreeClassifier()
    clf_dr = clf_dr.fit(train_X_DR,train_y_dr)

    if(i==5):
        clf_dr_5 = tree.DecisionTreeClassifier(max_depth=i)#max_depth=3)
        clf_dr_5 = clf_dr_5.fit(train_X_DR,train_y_dr)
        
        '''
    if(i < 10):
        vis_tree_dr = tree.export_graphviz(clf_dr, out_file=None, feature_names=names3_1,
                                        filled=True, rounded=True)
        graph_dr = graphviz.Source(vis_tree_dr)
        graph_dr.render("duration_tree_"+str(i))
        '''
    
    predictions_dr = (clf_dr.predict(test_X_DR))

    accuracy_dr = metrics.accuracy_score(predictions_dr,test_y_dr)
    depth_accuracies_dr.append(accuracy_dr)
    
    print("Accuracy with duration: ", accuracy_dr)
    print("Depth with ", i, " done!")


'''
axis_dr = []
for i in range(1,51):
    axis_dr.append(i)

plt.plot(axis_dr, depth_accuracies_dr)
plt.ylabel("Accuracy")
plt.xlabel("Tree Depth")
plt.savefig("plotting_dr.pdf")
plt.show()
'''


#################################################################
######################NOW WITH ALL######################
#################################################################
    
##TAILORING##
df_all = df2.set_axis(names, axis=1, inplace=False)

#df = df[df.job != "unknown"]
#df = df[df.default != "unknown"]
#df = df[df.poutcome != "nonexistent"]

#del df_all['duration']

le = preprocessing.LabelEncoder()

#print(df.loc[41182])
#print(df.loc[41159])
#print(df.loc[24013])

for column in df_all:
    df_all[column] = le.fit_transform(df_all[column])

names3_2 = ["age", "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week","durations","campaign", "pdays", "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
##TAILORING##

#print (df)
#print(df.loc[41182])
#print(df.loc[41159])
#print(df.loc[24013])

X_all = np.array(df_all.drop(['y'], 1))
y_all = np.array(df_all['y'])

train_X_all, test_X_all, train_y_all, test_y_all = train_test_split(X_all,y_all,test_size=0.2)

depth_accuracies_all = []
#depth_accuracies.append(0.750996015936255)

for i in range(1, 51):

    if (i < 50):
        clf_all = tree.DecisionTreeClassifier(max_depth=i)#max_depth=3)
    else:
        clf_all = tree.DecisionTreeClassifier()
    clf_all = clf_all.fit(train_X_all,train_y_all)

    if(i == 5):
        clf_all_5 = tree.DecisionTreeClassifier(max_depth=i)#max_depth=3)
        clf_all_5 = clf_all_5.fit(train_X_all,train_y_all)

        '''
        
    if (i < 10):
        vis_tree_all = tree.export_graphviz(clf_all, out_file=None, feature_names=names3_2,
                                            filled=True, rounded=True)
        graph_all = graphviz.Source(vis_tree_all)
        graph_all.render("all_tree_"+str(i))
        '''
    
    predictions = (clf_all.predict(test_X_all))

    accuracy = metrics.accuracy_score(predictions,test_y_all)
    depth_accuracies_all.append(accuracy)
    
    print("Accuracy with all: ", accuracy)
    print("Depth with ", i, " done!")

'''
axis_all = []
for i in range(1,51):
    axis_all.append(i)

plt.plot(axis_all, depth_accuracies_all)
plt.ylabel("Accuracy")
plt.xlabel("Tree Depth")
plt.savefig("plotting_all.pdf")
plt.show()
'''


#################################################################
######################NOW WITH ALL EXCEPT DURATIONS######################
#################################################################
    
##TAILORING##
df_all_e = df2.set_axis(names, axis=1, inplace=False)

#df = df[df.job != "unknown"]
#df = df[df.default != "unknown"]
#df = df[df.poutcome != "nonexistent"]

del df_all_e['duration']

le = preprocessing.LabelEncoder()

#print(df.loc[41182])
#print(df.loc[41159])
#print(df.loc[24013])

for column in df_all_e:
    df_all_e[column] = le.fit_transform(df_all_e[column])

names3_3 = ["age", "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week","campaign", "pdays", "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
##TAILORING##

#print (df)
#print(df.loc[41182])
#print(df.loc[41159])
#print(df.loc[24013])

X_all_e = np.array(df_all_e.drop(['y'], 1))
y_all_e = np.array(df_all_e['y'])

train_X_all_e, test_X_all_e, train_y_all_e, test_y_all_e = train_test_split(X_all_e,y_all_e,test_size=0.2)

#print(train_X_all_e.shape)
#print(test_X_all_e.shape)

depth_accuracies_all_e = []
#depth_accuracies.append(0.750996015936255)

for i in range(1, 51):

    if (i < 50):
        clf_all_e = tree.DecisionTreeClassifier(max_depth=i)#max_depth=3)
    else:
        clf_all_e = tree.DecisionTreeClassifier()
    clf_all_e = clf_all_e.fit(train_X_all_e,train_y_all_e)

    if(i == 5):
        clf_all_e_5 = tree.DecisionTreeClassifier(max_depth=i)#max_depth=3)
        clf_all_e_5 = clf_all_e_5.fit(train_X_all_e,train_y_all_e)

        '''
    if (i < 10):
        vis_tree_all_e = tree.export_graphviz(clf_all_e, out_file=None, feature_names=names3_3,
                                            filled=True, rounded=True)
        graph_all_e = graphviz.Source(vis_tree_all_e)
        graph_all_e.render("all_e_tree_"+str(i))
        '''
    predictions = (clf_all_e.predict(test_X_all_e))

    accuracy = metrics.accuracy_score(predictions,test_y_all_e)
    depth_accuracies_all_e.append(accuracy)
    
    print("Accuracy with all except duration: ", accuracy)
    print("Depth with ", i, " done!")

'''
axis_all_e = []
for i in range(1,51):
    axis_all_e.append(i)

plt.plot(axis_all_e, depth_accuracies_all_e)
plt.ylabel("Accuracy")
plt.xlabel("Tree Depth")
plt.savefig("plotting_all_e.pdf")
plt.show()
'''

#################################################################
######################NOW WITH MOST LOGICAL######################
#################################################################

##TAILORING##
df_ml = df2.set_axis(names, axis=1, inplace=False)

#df = df[df.job != "unknown"]
#df = df[df.default != "unknown"]
#df = df[df.poutcome != "nonexistent"]

del df_ml['duration']
del df_ml['day_of_week']

le = preprocessing.LabelEncoder()

#print(df.loc[41182])
#print(df.loc[41159])
#print(df.loc[24013])

for column in df_ml:
    df_ml[column] = le.fit_transform(df_ml[column])

names4 = ["age", "job", "marital", "education", "default", "housing", "loan", "contact", "month","campaign", "pdays", "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
##TAILORING##

#print (df)
#print(df.loc[41182])
#print(df.loc[41159])
#print(df.loc[24013])

X_ml = np.array(df_ml.drop(['y'], 1))
y_ml = np.array(df_ml['y'])

train_X_ml, test_X_ml, train_y_ml, test_y_ml = train_test_split(X_ml,y_ml,test_size=0.2)

depth_accuracies_ml = []
#depth_accuracies.append(0.750996015936255)

for i in range(1, 51):

    if (i < 50):
        clf_ml = tree.DecisionTreeClassifier(max_depth=i)#max_depth=3)
    else:
        clf_ml = tree.DecisionTreeClassifier()
    clf_ml = clf_ml.fit(train_X_ml,train_y_ml)

    if(i==5):
        clf_ml_5 = tree.DecisionTreeClassifier(max_depth=i)#max_depth=3)
        clf_ml_5 = clf_ml_5.fit(train_X_ml,train_y_ml)

        '''
    if (i < 10):
        vis_tree_ml = tree.export_graphviz(clf_ml, out_file=None, feature_names=names4,
                                           filled=True, rounded=True)
        graph_ml = graphviz.Source(vis_tree_ml)
        graph_ml.render("ml_tree_"+str(i))
        '''
    predictions = (clf_ml.predict(test_X_ml))

    accuracy = metrics.accuracy_score(predictions,test_y_ml)
    depth_accuracies_ml.append(accuracy)
    
    print("Accuracy with most logical: ", accuracy)
    print("Depth with ", i, " done!")

'''
axis_ml = []
for i in range(1,51):
    axis_ml.append(i)

plt.plot(axis_ml, depth_accuracies_ml)
plt.ylabel("Accuracy")
plt.xlabel("Tree Depth")
plt.savefig("plotting_ml.pdf")
plt.show()
'''

##Plotting altogether
axis = []
for i in range(1,51):
    axis.append(i)

plt.plot(axis, depth_accuracies, label="Highly Tailored", linewidth=4)
plt.plot(axis, depth_accuracies_dr, label="Higly Tailored with Duration",linewidth=4)
plt.plot(axis, depth_accuracies_all, label="All datapoints",linewidth=4)
plt.plot(axis, depth_accuracies_all_e, label="All datapoints w/o Duration",linewidth=4)
plt.plot(axis, depth_accuracies_ml, label= "Most logical cut",linewidth=4)
plt.ylabel("Accuracy")
plt.xlabel("Tree Depth")
#plt.savefig("plot_tog.pdf")
plt.legend(loc='lower left', framealpha=0.7)
#plt.legend()
plt.savefig("plot_tog.pdf")
plt.show()

important_5 = clf_5.feature_importances_
important_dr_5 = clf_dr_5.feature_importances_
important_all_5 = clf_all_5.feature_importances_
important_all_e_5 = clf_all_e_5.feature_importances_
important_ml_5 = clf_ml_5.feature_importances_

important = clf.feature_importances_
important_dr = clf_dr.feature_importances_
important_all = clf_all.feature_importances_
important_all_e = clf_all_e.feature_importances_
important_ml = clf_ml.feature_importances_

#print(names3_2)
#print(important)

plt.rcParams.update({'font.size': 12})

cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0,1,len(names)))

##TAILORED##
plt.figure()
plt.ylabel("Feature Importance")
plt.xlabel("Features")
plt.title("Importance of Features for Highly Tailored Data \n with depth=5")
plt.bar(names2,important_5,color=colors)
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.savefig("bar_tailored_5.pdf")
plt.show()

plt.figure()
plt.ylabel("Feature Importance")
plt.xlabel("Features")
plt.title("Importance of Features for Highly Tailored Data \n with depth=50")
plt.bar(names2,important,color=colors)
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.savefig("bar_tailored_50.pdf")
plt.show()
##TAILORED


##WITH DURATION##
plt.figure()
plt.ylabel("Feature Importance")
plt.xlabel("Features")
plt.title("Importance of Features for Highly Tailored Data with Duration \n with depth=5")
plt.bar(names3_1,important_dr_5,color=colors)
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.savefig("bar_dr_5.pdf")
plt.show()

plt.figure()
plt.ylabel("Feature Importance")
plt.xlabel("Features")
plt.title("Importance of Features for Highly Tailored Data with Duration \n with depth=50")
plt.bar(names3_1,important_dr,color=colors)
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.savefig("bar_dr_50.pdf")
plt.show()
##WITH DURATION##

##ALL##
plt.figure()
plt.ylabel("Feature Importance")
plt.xlabel("Features")
plt.title("Importance of Features for All Datapoints \n with depth=5")
plt.bar(names3_2,important_all_5,color=colors)
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.savefig("bar_all_5.pdf")
plt.show()

plt.figure()
plt.ylabel("Feature Importance")
plt.xlabel("Features")
plt.title("Importance of Features for All Datapoints \n with depth=50")
plt.bar(names3_2,important_all,color=colors)
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.savefig("bar_all_50.pdf")
plt.show()
##ALL##


##ALL EXCEPT DURATION##
plt.figure()
plt.ylabel("Feature Importance")
plt.xlabel("Features")
plt.title("Importance of Features for All Datapoints Except Duration \n with depth=5")
plt.bar(names3_3,important_all_e_5,color=colors)
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.savefig("bar_all_e_5.pdf")
plt.show()

plt.figure()
plt.ylabel("Feature Importance")
plt.xlabel("Features")
plt.title("Importance of Features for All Datapoints Except Duration \n with depth=50")
plt.bar(names3_3,important_all_e,color=colors)
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.savefig("bar_all_e_50.pdf")
plt.show()
##ALL EXCEPT DURATION##


##MOST LOGICAL CUT##
plt.figure()
plt.ylabel("Feature Importance")
plt.xlabel("Features")
plt.title("Importance of Features for Most Logical Cut \n with depth=5")
plt.bar(names4,important_ml_5,color=colors)
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.savefig("bar_ml_5.pdf")
plt.show()

plt.figure()
plt.ylabel("Feature Importance")
plt.xlabel("Features")
plt.title("Importance of Features for Most Logical Cut \n with depth=50")
plt.bar(names4,important_ml,color=colors)
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.savefig("bar_ml_50.pdf")
plt.show()
##MOST LOGICAL CUT##


