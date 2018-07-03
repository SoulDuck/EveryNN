f=open('./data_deleteme.csv','r')

pat_pred={}
pat_count={}
for line in f.readlines()[1:]:
    pat_code , pred_0 , pred_1 = line.split(',')[1:4]
    if not pat_code in pat_pred.keys(): # new
        pat_pred[pat_code]=float(pred_0)
        pat_count[pat_code] =1
    else:
        pat_pred[pat_code] += float(pred_0)
        pat_count[pat_code] += 1


for pat_code in pat_pred:
    print (pat_pred[pat_code] / pat_count[pat_code])
