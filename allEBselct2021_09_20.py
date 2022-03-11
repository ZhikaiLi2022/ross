from lightkurve import TessLightCurveFile
import lightkurve as lk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.optimize import curve_fit
import os
import scipy.stats
import scipy.special
import matplotlib
import matplotlib.mlab as mlab
from matplotlib import cm
import pandas as pd
import seaborn as sns
import pymc3 as pm
import celerite2
from celerite2 import terms

#----------------------------
#  在硬盘上匹配光变曲线数据  |
#----------------------------
def cross_ssd_data(ID): #ID的格式为["TIC 224292441"]
    path1 = [
    '/media/life/Seagate Expansion Drive/TESS数据下载/Sector1/',
    '/media/life/Seagate Expansion Drive/TESS数据下载/Sector2/',
    '/media/life/Seagate Expansion Drive/TESS数据下载/Sector3/',
    '/media/life/Seagate Expansion Drive/TESS数据下载/Sector4/',
    '/media/life/Seagate Expansion Drive/TESS数据下载/Sector5/',
    '/media/life/Seagate Expansion Drive/TESS数据下载/Sector6/',
    '/media/life/Seagate Expansion Drive/TESS数据下载/Sector7/',
    '/media/life/Seagate Expansion Drive/TESS数据下载2/Sector8/',
    '/media/life/Seagate Expansion Drive/TESS数据下载2/Sector9/',
    '/media/life/Seagate Expansion Drive/TESS数据下载2/Sector10/',
    '/media/life/Seagate Expansion Drive/TESS数据下载2/Sector11/',
    '/media/life/Seagate Expansion Drive/TESS数据下载2/Sector12/',
    '/media/life/Seagate Expansion Drive/TESS数据下载2/Sector13/',
    ]
    file_dir1 = [[i+j for j in os.listdir(i)] for i in path1]
    path2 = [
    '/media/life/Seagate Expansion Drive/TESS数据下载/Sector14/',
    '/media/life/Seagate Expansion Drive/TESS数据下载2/Sector15/',
    '/media/life/Seagate Expansion Drive/TESS数据下载/Sector16/',
    '/media/life/Seagate Expansion Drive/TESS数据下载2/Sector17/',
    '/media/life/Seagate Expansion Drive/TESS数据下载/Sector18/',
    '/media/life/Seagate Expansion Drive/TESS数据下载2/Sector19/',
    '/media/life/Seagate Expansion Drive/TESS数据下载/Sector20/',
    '/media/life/Seagate Expansion Drive/TESS数据下载/Sector21/',
    '/media/life/Seagate Expansion Drive/TESS数据下载2/Sector22/',
    '/media/life/Seagate Expansion Drive/TESS数据下载2/Sector23/',
    '/media/life/Seagate Expansion Drive/TESS数据下载2/Sector24/',
    '/media/life/Seagate Expansion Drive/TESS数据下载2/Sector25/',
    '/media/life/Seagate Expansion Drive/TESS数据下载2/Sector26/',
    ]
    file_dir2 = [[i+j for j in os.listdir(i)] for i in path2]
    file_dir = file_dir1+file_dir2
    IDdir = []
    for i in range(len(ID)):
        IDdir.append([])
    for i in range(len(ID)):
            for k in range(len(file_dir)):
                for l in range(len(file_dir[k])):
                    a = ID[i][4:]
                    if a == file_dir[k][l][-len(a)-15:-15] and file_dir[k][l][-len(a)-16:-len(a)-15] == '0':
                        IDdir[i].append(file_dir[k][l])
                        continue
            print("IDdir--"+str(i))
    return IDdir

#-------------
#  冒泡排序  |
#-------------
def bubble_sort(arr):
    if arr is None and len(arr) < 2:
        return
    for end in range(len(arr) - 1, -1, -1):
        exchange = False  # 设置哨兵
        for i in range(end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                exchange = True
        if not exchange:
            return arr
    return arr

#---------------------------
#  与公式计算值作对比求出e  |
#---------------------------
def e(p_opt,p_opt1):
    t1 = abs(p_opt[1]-p_opt1[1])
    if t1>0.5:
        t1 = 1-t1
    e = np.arange(0,1,0.00001)
    result = []
    for i in e:
        t2 = (1/np.pi) * ( np.arccos(i) - i*(1-i**2)**(1/2) )
        result.append(t2)
    e1 = [abs(t1-i) for i in result]
    e2 = e1.index(min(e1))/100000
    return e2 

#----------------
#  定义GSR函数  |
#----------------
def GPR(x, y, yerr, X_n):
    term1 = terms.SHOTerm(sigma=1.0, rho=1.0, tau=10.0)
    term2 = terms.SHOTerm(sigma=1.0, rho=5.0, Q=0.25)
    kernel = term1 + term2
    X_new = np.linspace(-0.5, 0.5, X_n)
    gp = celerite2.GaussianProcess(kernel, mean=0.0)
    gp.compute(x,yerr = yerr)
    print("Initial log likelihood: {0}".format(gp.log_likelihood(y)))
    mu, variance = gp.predict(y, X_new, return_var=True)
    sigma = np.sqrt(variance)
    return mu, sigma, X_new    

#-------------
#  差分计算  |
#------------- 
def difference(mu):
    cha = [mu[i]-mu[i-1] for i in range(1,len(mu))]
    min_time_power = []
    for i in range(1,len(cha)):
        if cha[i]/abs(cha[i]) == -cha[i-1]/abs(cha[i-1]):                
            if cha[i] > 0:
                min_time_power.append(mu[i])
    bubble_sort(min_time_power) 
    min_time= [] 
    for i in min_time_power:  #if i < 1-(1-min_time_power[0])*0.05:
        min_time.append(list(mu).index(i))     
    return cha, min_time 
        
#------------
#   取峰    |      
#------------
'''
lim1的作用是在一定范围(0.05)内寻峰,如果=None就不限定范围。
sub是第一个峰min_time1在min_time的序列位置，去除这个峰选取另一个作为第二个峰
'''
def remove_peak(X_new,mu,cha,fold1,min_time,lim1, sub):   
    if lim1 !=None:   #差0.5相位处左右0.05flux最低的。 
        result = [abs(lim1-X_new[i]) for i in min_time]
        result1 = []
        for i in result:
            if i < 0.05:
                result1.append(i)
        min_time1 = min_time[result.index(min(result1))]
    else: min_time1 = min_time[0]
    if sub != None:
        if sub == 0:
            min_time1 = min_time[1]
        else:
            min_time1 = min_time[0]
    X_new[min_time1] #主极小时刻在X_new的位置
    #求掩食star和end的位置
    star1_p = 0 #如果是没有下降的一个就用第一个点就可以了。 
    for i in range(min_time1,1,-1):
        h_c1 = mu[i]-mu[i-1]
        if h_c1 > 0:
            star1_p = i
            break  
    end1_p = len(X_new)-1 #如果是没有下降的一个就用最后一个点就可以了。 
    for i in range(min_time1,len(X_new)-1):
        q_c1 = mu[i]-mu[i+1]
        if q_c1 > 0:
            end1_p = i
            break  
    #得到正确的star和end在X_new的位置
    true_star1 = X_new[star1_p]
    true_end1 = X_new[end1_p]
    #将掩食部分data截取下来 
    ss1 = []
    for i in range(len(fold1.phase.value)):
        ss1.append(abs(fold1.phase.value[i]-true_star1))
    s1 = ss1.index(min(ss1))
    ee1 = []   
    for i in range(len(fold1.phase.value)):
        ee1.append(abs(fold1.phase.value[i]-true_end1))
    e1 = ee1.index(min(ee1))
    x_f = fold1.phase.value[s1:e1]
    y_f = fold1.flux.value[s1:e1]
    return x_f, y_f, true_star1, true_end1, min_time1

#----------------
# 拟合取出的数据 |
#----------------
'''
    print(p_opt)
    print('Amplitude: {} +\- {}'.format(p_opt[0], np.sqrt(p_cov[0,0])))
    print('Mean: {} +\- {}'.format(p_opt[1], np.sqrt(p_cov[1,1])))
    print('Standard Deviation: {} +\- {}'.format(p_opt[2], np.sqrt(p_cov[2,2])))
'''
def GSfit(x_f,y_f,X_new,min_time1 ):
    yy = [i for i in y_f]
    def f(x,a,b,c,d):
        return a * np.exp(-(x-b)**2/(2.0*c**2))+d
    param_bounds = ([-1,round(X_new[min_time1]-0.05,2),0,0],[0,round(X_new[min_time1]+0.05,2),np.inf,np.inf])
    p_opt, p_cov = scipy.optimize.curve_fit(f,x_f, yy,maxfev = 10000,bounds=param_bounds)
    a,b,c,d = p_opt
    best_fit_gauss_2 = f(x_f,a,b,c,d)
    return p_opt, best_fit_gauss_2

'''
#-------------
#  找最低谷  |
#-------------
def find_ebb(fold_phase,fold_flux):
    a = [i for i in fold_flux ]
    g = [i for i in fold_phase]  
    result3 = [] 
    result4 = [] 
    d = min(a) 
    for i in range(100): 
        result1 = [] 
        result2 = [] 
        for j in range(len(a)): 
            if a[j] - d > 0: 
                result1.append(a[j]-d) 
                result2.append(j) 
        e = result2[result1.index(min(result1))]    
        d = a[e] 
        result3.append(d)  
        result4.append(g[e])
    return result3, result4 
'''
#------------------------------------------------------------------------------------------------------|
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  好戏刚刚开始!!  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
#------------------------------------------------------------------------------------------------------|
#-------------
#  IDdir获取 |
#-------------
'''
#如果需要读取文件名字请启用这块代码
path1 = "/home/life/双星偏心率的求解/allEB1-13/"
path2 = "/home/life/双星偏心率的求解/allEB14-26/"
line1 = os.listdir(path1)
line2 = os.listdir(path2)
for i in range(len(line1)):
    if "f" in line1[i]:
        line1[i] = line1[i][1:]
line = [i[:-4] for i in line1+line2]
IDdir = cross_ssd_data(line)
'''
path = "/home/life/双星偏心率的求解/allEBselct/default_period.txt"
f = open(path,'r');line = f.readlines();f.close()
IDdir = cross_ssd_data([i[:18].strip() for i in line])
#--------------------
# 把求好的周期取出来 |
#--------------------
path = "/home/life/双星偏心率的求解/allEBselct/default_period.txt"
f = open(path,'r');line = f.readlines();f.close()
period = [round(float(i[20:-1].strip()),6) for i in line]
#---------------
# 开始执行流程 |
#---------------
result1 = open("/home/life/双星偏心率的求解/allEBselct/e_data_picture/1/e_result.txt",'a')
result2 = open("/home/life/双星偏心率的求解/allEBselct/e_data_picture/2/e_result.txt",'a')
result3 = open("/home/life/双星偏心率的求解/allEBselct/e_data_picture/4/e_result.txt",'a')
result4 = open("/home/life/双星偏心率的求解/allEBselct/e_data_picture/8/e_result.txt",'a')
result5 = open("/home/life/双星偏心率的求解/allEBselct/e_data_picture/16/e_result.txt",'a')
result6 = open("/home/life/双星偏心率的求解/allEBselct/e_data_picture/random/e_result.txt",'a')
P_maxp8  = []#里面放的是 P_maxp0.5 的圆双星
P_maxp1  = []#里面放的是 P_maxp0.5的偏心双星和 P_maxp1的圆双星
P_maxp2  = []#里面放的是 P_maxp1的偏心双星和 P_maxp2的圆双星
P_maxp4  = []#里面放的是 P_maxp2的偏心双星和 P_maxp4的圆双星
P_maxp8  = []#里面放的是 P_maxp4的偏心双星和 P_maxp8的圆双星
P_maxp16 = []#里面放的是 P_maxp8的偏心双星和 P_maxp16的圆双星
err_GS   = []#存放GS取峰出错的星

for k in [2,4,8,16]:#循环过后，IDdir中留下的是P_maxp4 的偏心双星
    err_ID1 = [] #存放有些打不开的fits文件
    for ii in range(len(IDdir)):
        dirall = []
        for j in IDdir[ii]:
            try:
                dirall.append(lk.TessLightCurveFile(j))
            except:
                err_ID1.append(j)
                print(j)
        if dirall == []:
            continue
        if len(dirall) > 5:
             dirall = dirall[:5]
        target = "random"
        lc0 = lk.LightCurveCollection(dirall)
        lc = lc0.stitch().remove_nans()
        max_period = k*period[ii]
        bin_n = (2000)/len(lc.bin(1)) 
        lcc =  lc.bin(round(1/bin_n,5)).remove_nans()   
        fold = lcc.fold(max_period,normalize_phase=True,epoch_time=lcc.time.value[list(lcc.flux.value).index(min(lcc.flux.value))]-0.40*max_period)
        fold1 = lc.fold(max_period,normalize_phase=True,epoch_time=lcc.time.value[list(lcc.flux.value).index(min(lcc.flux.value))]-0.40*max_period)
        x = fold.phase.value
        y = fold.flux.value
        yerr = fold.flux_err.value
        minimum = 'unknown'
        #----------------------------------------------------------------------------------------|
        #   fold为bin后的data，fold1没有bin                                                       |
        #   X_new 为GPR横轴，mu为GPR纵轴                                                          |
        #   cha为差分y轴数据                                                                      |
        #   x_f,y_f,x1_f,x1_f为取出的峰的数据，p_opt中记录着GS拟合的结果                           |
        #   true_star,true_end为掩食开始和结束的相位                                               |
        #   min_time为位置数列，记录波动在X_new的序列位置，min_time+n为选择的峰的在X_new的序列位置   |
        #-----------------------------------------------------------------------------------------|
        try:
            mu, sigma, X_new = GPR(x, y, yerr, 200)#GPR结果
            cha, min_time = difference(mu)#差分寻峰结果
            x_f  , y_f  , true_star1, true_end1, min_time1 = remove_peak(X_new,mu,cha,fold1,min_time,0.4,sub=None)   #取峰
            p_opt, best_fit_gauss_2 = GSfit(x_f,y_f,X_new,min_time1 ) #先把第一个峰拟合出来
            x1_f , y1_f , true_star2, true_end2, min_time2 = remove_peak(X_new,mu,cha,fold1,min_time,lim1=None,sub=min_time.index(min_time1))#取另一个峰 
            p_opt1, best_fit_gauss_21 = GSfit(x1_f,y1_f,X_new,min_time2)
            minimum = 'no equal'
        except:
            err_GS.append(lc.label)
            continue
        #------------------------------------------------------------------------------------------
        # 加个判断，看看0.5相位是否有个相同的（神来之笔）|
        #------------------------------------------------------------------------------------------
        try:
            x_f_sub  , y_f_sub  , true_star1_sub, true_end1_sub, min_time1_sub = remove_peak(X_new,mu,cha,fold1,min_time,p_opt[1]-0.5,sub=None)   #取峰
            p_opt_sub, best_fit_gauss_2_sub = GSfit(x_f_sub,y_f_sub,X_new,min_time1_sub )#看看0.5相位差处的峰的高度
            p_opt[3]+p_opt[0]#最低点的y值
            p_opt[0]*0.05
            if p_opt[3]+p_opt[0]-abs(p_opt[0]*0.16) < p_opt_sub[3]+p_opt_sub[0] < p_opt[3]+p_opt[0]+abs(p_opt[0]*0.16):
                x1_f , y1_f , true_star2, true_end2, min_time2 = x_f_sub  , y_f_sub  , true_star1_sub, true_end1_sub, min_time1_sub
                p_opt1, best_fit_gauss_21 = p_opt_sub, best_fit_gauss_2_sub
                #挖除主极小范围内点，之后的序列定义为min_time_new
                min_time_new = []
                for i in min_time:
                    if true_star1 < X_new[i] < true_end1 or true_star2 < X_new[i] < true_end2:
                        print('a')
                    else: min_time_new.append(i)
                x_f_c  , y_f_c  , true_star1_c, true_end1_c, min_time1_c = remove_peak(X_new,mu,cha,fold1,min_time_new,lim1=None,sub=None)  #尝试取出次极小 
                p_opt_c, best_fit_gauss_2_c = GSfit(x_f_c,y_f_c,X_new,min_time1_c ) #先把第一个峰拟合出来
                x1_f_c , y1_f_c , true_star2_c, true_end2_c, min_time2_c = remove_peak(X_new,mu,cha,fold1,min_time_new,lim1=None,sub=min_time_new.index(min_time1_c))#取另一个次极小峰
                p_opt1_c, best_fit_gauss_21_c = GSfit(x1_f_c,y1_f_c,X_new,min_time2_c)
                EEB = []
                for i in cha:
                    if abs(i) < 0.0124747165*(max(mu)-min(mu)):
                        EEB.append(i) 
                present = len(EEB)/199
                if p_opt1[1]<p_opt_c[1]<p_opt[1]:
                    phase1 = p_opt[1]-p_opt_c[1]
                else: 
                    phase1 = p_opt1[1]-p_opt_c[1]
                if present > 0.5:
                    if phase1 > 0.253 or phase1< 0.247:
                        if p_opt_c[3]+p_opt_c[0]-abs(p_opt_c[0]*0.20) < p_opt1_c[3]+p_opt1_c[0] < p_opt_c[3]+p_opt_c[0]+abs(p_opt_c[0]*0.20):
                            if 0.49< abs(p_opt_c[1]-p_opt1_c[1]) < 0.51:                                                
                                minimum = 'equal'
            
        except:
            print("no mean")
        #------------------------------------------------------------------------------------------        
        eccentric = e(p_opt,p_opt1)
        if k == 1:
            if eccentric < 0.01:
                P_maxp1 .append(IDdir[ii])
                IDdir[ii] = []
                target = "EBP_1"     
                result1.write("{: <20}".format(lc.label))
                result1.write("{: <20}".format(lc.ra))
                result1.write("{: <20}".format(lc.dec))
                result1.write("{: <20}".format(str(max_period)))
                result1.write("{: <20}".format(str(eccentric)))
                result1.write("\n")
        if k == 2:
            if eccentric < 0.01:
                P_maxp2.append(IDdir[ii])
                IDdir[ii] = []
                target = "EBP_2 and EEBP_1"
                result2.write("{: <20}".format(lc.label))
                result2.write("{: <20}".format(lc.ra))
                result2.write("{: <20}".format(lc.dec))
                result2.write("{: <20}".format(str(max_period)))
                result2.write("{: <20}".format(str(eccentric)))
                result2.write("\n")
        if k == 4:
            if eccentric < 0.01:
                P_maxp4.append(IDdir[ii])
                IDdir[ii] = []
                target = "EBP_4 and EEBP_2"
                result3.write("{: <20}".format(lc.label))
                result3.write("{: <20}".format(lc.ra))
                result3.write("{: <20}".format(lc.dec))
                result3.write("{: <20}".format(str(max_period)))
                result3.write("{: <20}".format(str(eccentric)))
                result3.write("\n")
        if k == 8:
            if eccentric < 0.01:
                P_maxp8.append(IDdir[ii])   
                IDdir[ii] = []  
                target = "EBP_8 and EEBP_4"      
                result4.write("{: <20}".format(lc.label))
                result4.write("{: <20}".format(lc.ra))
                result4.write("{: <20}".format(lc.dec))
                result4.write("{: <20}".format(str(max_period)))
                result4.write("{: <20}".format(str(eccentric)))
                result4.write("\n")
        if k == 16:
            if eccentric < 0.01:
                P_maxp16.append(IDdir[ii])   
                IDdir[ii] = []  
                target = "EBP_16 and EEBP_8"      
                result5.write("{: <20}".format(lc.label))
                result5.write("{: <20}".format(lc.ra))
                result5.write("{: <20}".format(lc.dec))
                result5.write("{: <20}".format(str(max_period)))
                result5.write("{: <20}".format(str(eccentric)))
                result5.write("\n")
        #绘图展示数据
        fig = plt.figure(figsize=(13,11))
        gs = gridspec.GridSpec(4,6)
        ax3 = plt.subplot(gs[0,:])
        lc.scatter(ax=ax3,label='',c='k',s=1)
        plt.title(lc.label+"e={}".format(str(eccentric)),fontsize=15)
        ax3.tick_params(which='both',labelsize=10,direction='in',top=True, right=True)
        ax3.text(.020, .7, '(a)',
                            horizontalalignment='left',
                            transform=ax3.transAxes, fontsize=15)
        ax4 = plt.subplot(gs[1:3,0:2])
        ax4.tick_params(which='both',labelsize=10,direction='in',top=True, right=True)
        ax4.text(.020, .7, '(b)',
                            horizontalalignment='left',
                            transform=ax4.transAxes, fontsize=15)
        ax2 = plt.subplot(gs[1,2:])
        plt.plot(X_new[1:], cha,c='k')
        plt.scatter(X_new[1:], cha,c='',edgecolors='k')
        plt.axvline(true_star1,c='r',ls='--',lw=1)
        plt.axvline(true_star2,c='r',ls='--',lw=1)
        plt.axvline(true_end1+(X_new[1]-X_new[0]),c='r',ls='--',lw=1)
        plt.axvline(true_end2+(X_new[1]-X_new[0]),c='r',ls='--',lw=1)
        ax2.set_xlabel('Phase',fontsize=10)
        ax2.set_ylabel('Difference',fontsize=10)
        ax2.tick_params(which='both',labelsize=10,direction='in',top=True, right=True)
        ax2.text(.020, .7, '(c)',
                            horizontalalignment='left',
                            transform=ax2.transAxes, fontsize=15)
        ax1 = plt.subplot(gs[2,2:])
        fold1.scatter(ax=ax1,s=0.1,c='gray',label='')
        plt.scatter(X_new, mu ,c='',edgecolors='k', label="GPR");
        plt.axvline(true_star1,c='r',ls='--',lw=1)
        plt.axvline(true_star2,c='r',ls='--',lw=1)
        if minimum == 'equal':
            plt.axvline(true_star1_c,c='r',ls='--',lw=1)
            plt.axvline(true_end1_c,c='r',ls='--',lw=1)
            plt.axvspan(true_star1_c,true_end1_c,facecolor='r',alpha=0.3) 
        plt.axvspan(true_star1,true_end1,facecolor='gray',alpha=0.3)
        plt.axvline(true_end1,c='r',ls='--',lw=1)
        plt.axvline(true_end2,c='r',ls='--',lw=1)
        if minimum == 'equal':
            plt.axvline(true_star2_c,c='r',ls='--',lw=1)
            plt.axvline(true_end2_c,c='r',ls='--',lw=1) 
            plt.axvspan(true_star2_c,true_end2_c,facecolor='r',alpha=0.3) 
        plt.axvspan(true_star2,true_end2,facecolor='gray',alpha=0.3) 
        ax1.set_xlabel('Phase',fontsize=10)
        ax1.set_ylabel('Normalized flux',fontsize=10)
        ax1.tick_params(which='both',labelsize=10,direction='in',top=True, right=True)
        ax1.text(.020, .7, '(d)',
                            horizontalalignment='left',
                            transform=ax1.transAxes, fontsize=15)    
        plt.legend()    
        ax5 = plt.subplot(gs[3,0:3])
        plt.scatter(x_f,y_f,s=1,c='gray')
        plt.plot(x_f, [i for i in best_fit_gauss_2],'r')
        plt.axvline(true_star1,c='r',ls='--',lw=1)
        plt.axvline(true_end1,c='r',ls='--',lw=1)               
        plt.axvline(p_opt[1],c='k',ls='--',lw=1)
        ax5.set_xlabel('Phase',fontsize=10)
        ax5.set_ylabel('Normalized flux',fontsize=10)
        plt.title('Primary minimum',fontsize=10)
        ax5.tick_params(which='both',labelsize=10,direction='in',top=True, right=True)
        ax5.text(.010, .7, '(e)',
                            horizontalalignment='left',
                            transform=ax5.transAxes, fontsize=15)
        ax6 = plt.subplot(gs[3,3:])
        plt.scatter(x1_f,y1_f,s=1,c='gray')
        plt.plot(x1_f, [i for i in best_fit_gauss_21],'r')
        plt.axvline(true_star2,c='r',ls='--',lw=1)
        plt.axvline(true_end2,c='r',ls='--',lw=1)
        plt.axvline(p_opt1[1],c='k',ls='--',lw=1)
        ax6.set_xlabel('Phase',fontsize=10)
        ax6.set_ylabel('Normalized flux',fontsize=10)
        plt.title('Subminimum',fontsize=10)
        ax6.tick_params(which='both',labelsize=10,direction='in',top=True, right=True)
        ax6.text(.010, .7, '(f)',
                            horizontalalignment='left',
                            transform=ax6.transAxes, fontsize=15)
        plt.subplots_adjust(top=0.960,bottom=0.05,left=0.09,right=0.970,hspace=0.400,wspace=0.635)
        if target == "random":
            plt.savefig("/home/life/双星偏心率的求解/allEBselct/e_data_picture/random/"+str(k)+lc.label+".png")
            result6.write("{: <20}".format(lc.label))
            result6.write("{: <20}".format(lc.ra))
            result6.write("{: <20}".format(lc.dec))
            result6.write("{: <20}".format(str(max_period)))
            result6.write("{: <20}".format(str(eccentric)))
            result6.write("\n")
        if target == "EBP_1":
            if minimum == 'no equal' or minimum == 'unknown':
                plt.savefig("/home/life/双星偏心率的求解/allEBselct/e_data_picture/1/"+str(k)+lc.label+".png")
            if minimum == 'equal':
                plt.savefig("/home/life/双星偏心率的求解/allEBselct/e_data_picture/1/EEB/"+str(k)+lc.label+".png")
        if target == "EBP_2 and EEBP_1":
            if minimum == 'no equal' or minimum == 'unknown':
                plt.savefig("/home/life/双星偏心率的求解/allEBselct/e_data_picture/2/"+str(k)+lc.label+".png")
            if minimum == 'equal':
                plt.savefig("/home/life/双星偏心率的求解/allEBselct/e_data_picture/2/EEB/"+str(k)+lc.label+".png")
        if target == "EBP_4 and EEBP_2":
            if minimum == 'no equal' or minimum == 'unknown':
                plt.savefig("/home/life/双星偏心率的求解/allEBselct/e_data_picture/4/"+str(k)+lc.label+".png")
            if minimum == 'equal':
                plt.savefig("/home/life/双星偏心率的求解/allEBselct/e_data_picture/4/EEB/"+str(k)+lc.label+".png")
        if target == "EBP_8 and EEBP_4" :
            if minimum == 'no equal' or minimum == 'unknown':
                plt.savefig("/home/life/双星偏心率的求解/allEBselct/e_data_picture/8/"+str(k)+lc.label+".png")
            if minimum == 'equal':
                plt.savefig("/home/life/双星偏心率的求解/allEBselct/e_data_picture/8/EEB/"+str(k)+lc.label+".png")
        if target == "EBP_16 and EEBP_8" :
            if minimum == 'no equal' or minimum == 'unknown':
                plt.savefig("/home/life/双星偏心率的求解/allEBselct/e_data_picture/16/"+str(k)+lc.label+".png")  
            if minimum == 'equal':
                plt.savefig("/home/life/双星偏心率的求解/allEBselct/e_data_picture/16/EEB/"+str(k)+lc.label+".png")  
        plt.close('all')
        print(str(k)+"倍折叠的第"+str(ii)+"个源处理完毕")    
