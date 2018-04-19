f=open('trainLabels.csv' , 'r')
lines=f.readlines()
for line in lines:
    print line.split(',')[-1]