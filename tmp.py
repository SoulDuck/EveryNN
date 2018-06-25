import os , glob
import numpy as np
import csv
import Tester
"""

paths_0=glob.glob('./images/0100-0000003-019_label_0/*.png')
paths_0=sorted(paths_0)
paths_1=glob.glob('./images/0100-0000003-019_label_1/*.png')
paths_1=sorted(paths_1)
imgs= []
# Label
n_label_0=len(paths_0)
n_label_1=len(paths_1)
print (n_label_1)
print (n_label_0)
test_labs=np.zeros([n_label_0 + n_label_1 , 2])
test_labs[:n_label_0 , 0]=1
test_labs[n_label_0 : ,1 ]=1

test_cls=np.argmax(test_labs , axis=1)
"""
f=open('/Users/seongjungkim/Desktop/prediction_cac_row.csv','r')
ret_dict={}
count =0
ret_cls={}

for line in f.readlines()[1:]:
    fname , pred_0 , pred_1  ,test_cls = line.split(',')[1:5]
    # name
    fname=os.path.splitext(fname)[0]
    pred_1=pred_1.replace(']', '')
    try:
        pred_0 , pred_1 =map(float , [pred_0 , pred_1])
    except :
        print fname, pred_0, pred_1
    # binding the Same patients together
    fname = fname.replace('_L', '').replace('_R', '')
    if not fname in ret_dict.keys():
        ret_dict[fname]=[pred_0]
        ret_cls[fname] = int(test_cls)
    else:
        ret_dict[fname].append(pred_0)
    #print fname, pred_0, pred_1, test_cls
    count += 1
assert len(ret_cls) == len(ret_dict)
f.close()

f=open('/Users/seongjungkim/Desktop/prediction_cac_.csv','w')
writer=csv.writer(f)

means=[]
patient_labels=[]
labels = []
for key in ret_dict.keys():
    mean = np.mean(ret_dict[key])
    writer.writerow([key , mean])
    means.append(mean)
    labels.append(ret_cls[key])


means=np.asarray(means)
indices=np.where([means >= 0.5])[1]
rev_indices=np.where([means < 0.5])[1]

print means
print ret_cls

tester=Tester.Tester(None)
tester.plotROC( means , labels , 'Patient ROC curve' , savepath='tmp.png')
