from sklearn import tree 
import pandas as pd 
import pydotplus
import numpy as np
from sklearn.model_selection  import  train_test_split 
from IPython.display import Image
from sklearn import tree, metrics
TeamAnalysis_df = pd.read_csv('m.csv',names=['Home-Away','FG','FGA','FGY','3P','3PA','3PY','FT','FTA','FTY','ORB','DRB','AST','STL','BLK','TOV','PF','Score'])
TeamAnalysis_df.head()
TeamAnalysis_df['Score'],class_names = pd.factorize(TeamAnalysis_df['Score'])
print(class_names)
print(TeamAnalysis_df['Score'].unique())
X = TeamAnalysis_df.iloc[:,:-1]
y = TeamAnalysis_df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)
dtree = tree.DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=0)
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))
feature_names = X.columns
dot_data = tree.export_graphviz(dtree, out_file=None, filled=True, rounded=True,
                                feature_names=feature_names,  
                                class_names=class_names)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png()) 
graph.write_png("m1(ent).png")