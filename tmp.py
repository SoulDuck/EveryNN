f=open('trainLabels.csv' , 'r')
lines=f.readlines()
count_0 , count_1 , count_2 , count_3 =0,0,0,0
for line in lines:
    if line.split(',')[-1] ==0:
        count_0+=1
    elif line.split(',')[-1] ==1:
        count_1 += 1
    elif line.split(',')[-1] == 2:
        count_2 += 1
    elif line.split(',')[-1] == 3:
        count_3 += 1
print count_0 ,count_1,count_2,count_3