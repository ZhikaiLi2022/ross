import os
path1 = "/home/life/双星偏心率的求解/汇总2021_9_30/new_EEB.txt" 
path2 = "/home/life/双星偏心率的求解/汇总2021_9_30/table_199stars.txt"
path3 = "/home/life/双星偏心率的求解/allEBselct/e_data_picture/all EEB/new_can/"
f1 = open(path1,'r');line1 = f1.readlines();f1.close()
f2 = open(path2,'r');line2 = f2.readlines();f2.close()
line3 = [i[1:-4] for i in os.listdir(path3)]
new = [i[:20].rstrip(" ") for i in line1+line2]
code = set(line3)
view = set(new)
same = view & code 

path = "/home/life/双星偏心率的求解/code/"
f1 = open(path+'cross.txt','r')
f2 = open(path+'Justesen2021.dat','r')
f3 = open(path+'Kjurkchieva.txt','r')
f4 = open(path+'529ebkepler.txt','r')
f5 = open(path+'table.txt','a')
f6 = open(path+'Kjurkchieva2017.txt','r')
line11 = f1.readlines();f1.close()
line21 = f2.readlines();f2.close()
line31 = f3.readlines();f3.close()
line41 = f4.readlines();f4.close()
line61 = f6.readlines();f6.close()
juTIC = ['TIC '+i[:9].lstrip() for i in line21[21:]]
our = set(new)
other = set(juTIC)
same = our & other 
juecosw = [float(i[58:73]) for i in line21[21:]]
jubig = []
for i in range(len(juecosw)):
    if juecosw[i] >= 0.01:
        jubig.append(juTIC[i])
big = set(jubig)
s = big & same #91
o = big - s    #51
path4 = "/home/life/双星偏心率的求解/allEB1-13/"
path5 = "/home/life/双星偏心率的求解/allEB14-26/"
line4 = os.listdir(path4)
line5 = os.listdir(path5)
for i in range(len(line4)):
    if "f" in line4[i]:
        line4[i] = line4[i][1:]
line6 = [i[:-4] for i in line4+line5]
all_our = set(line6) #5448
alls = o & all_our   #19
