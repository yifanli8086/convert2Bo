import os, sys
filedir = os.getcwd()+'/results'
#filenames=os.listdir(filedir)
f=open('result.txt','w')
filename = ''
num = int('1')
for num in range (1, int(sys.argv[1])):
    if (num < 10): 
    	filename = '000' + str(num) + '.txt'
    elif (num < 100):
    	filename = '00' + str(num) + '.txt'
    elif (num < 1000):
    	filename = '0' + str(num) + '.txt'
    else:
    	filename = str(num) + '.txt'
    filepath = filedir+'/'+filename
    for line in open(filepath):
        f.writelines(str(num) + " " + line)
    num += 1
f.close()
