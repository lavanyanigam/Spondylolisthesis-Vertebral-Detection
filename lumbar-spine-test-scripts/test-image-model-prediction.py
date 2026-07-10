import os
import csv
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


true_label = "/Users/lavanyanigam/Desktop/spondylolisthesis-vertebral-project/lumbar-spine-test-labels/lumbar-spine-test-label-actual.csv"
predicted_label= "/Users/lavanyanigam/Desktop/spondylolisthesis-vertebral-project/lumbar-spine-test-labels/lumbar-spine-test-label-predicted.csv"

tl= pd.read_csv(true_label).set_index("Image_Name")
pl=pd.read_csv(predicted_label).set_index("Image_Name")

common=tl.index.intersection(pl.index)
correct=(tl.loc[common,"Diagnosis"]==pl.loc[common,"Diagnosis"]).sum()
total=len(common)

accuracy_perct=(correct/total)*100     
labels_order = ['Normal', 'Anterolisthesis', 'Retrolisthesis'] 
print("Accuracy: ", accuracy_perct)
cm=confusion_matrix(tl.loc[common,"Diagnosis"],pl.loc[common,"Diagnosis"], labels=labels_order)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_order )
disp.plot(cmap=plt.cm.Blues,values_format='d')
plt.show()

