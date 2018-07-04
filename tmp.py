import Tester
import numpy as np
f=open('./pred_list_2.csv','r')

pat_pred={}
pat_count={}
pat_label={}
for line in f.readlines()[1:]:
    pat_code , pred_0 , pred_1 , label = line.split(',')[1:5]
    if not pat_code in pat_pred.keys(): # new
        pat_pred[pat_code]=float(pred_0)
        pat_count[pat_code] = 1
        if label.strip() == '1':
            pat_label[pat_code] = 0
            pass;
        else:
            pat_label[pat_code] = 1

    else:
        pat_pred[pat_code] += float(pred_0)
        pat_count[pat_code] += 1

preds=[]
labels=[]
for pat_code in pat_pred:
    print (pat_pred[pat_code] / pat_count[pat_code]) , pat_label[pat_code]
    preds.append(pat_pred[pat_code])
    labels.append(pat_label[pat_code])

print type(preds[0])
print len(preds)
print len(labels)
preds=np.asarray(preds)
labels =np.asarray(labels)
tester=Tester.Tester(None)
tester.plotROC(predStrength=preds, labels=labels ,prefix='cac ensemble ROC Curve' , savepath='ensemble_cac.png')