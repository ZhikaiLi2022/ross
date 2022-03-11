import os
path1 = "/home/life/双星偏心率的求解/allEBselct-old/newEEBperiod/default_period.txt"
path2 = "/home/life/双星偏心率的求解/allEBselct/e_data_picture/all EEB/newEEBperiod/default_period.txt"
path3 = "/home/life/双星偏心率的求解/e_data_picture1/can not/"
f1 = open(path1,'r');line1 = f1.readlines();f1.close()
f2 = open(path2,'r');line2 = f2.readlines();f2.close()
line3 = [i[1:-4] for i in os.listdir(path3)]
new1 = [i[:20].rstrip(" ") for i in line1+line2]
period1 = [round(float(i[20:-1]),5) for i in line1+line2]
all = []
for i in line3:
    for j in line1+line2:
        if i in j:
            all.append(j)
'''
#第次缩小X_new剩下的源20个
all1 = 
['TIC 141525324       0.72492225          \n',
 'TIC 70999666        2.9441816470076554  \n',
 'TIC 349154435       4.432554606486438   \n',
 'TIC 198366673       1.7599382257933158  \n',
 'TIC 200297691       6.979710342299365   \n',
 'TIC 240122720       0.26939071          \n',
 'TIC 318986273       3.107804023336993   \n',
 'TIC 419744996       7.003968921043849   \n',
 'TIC 20299889        7.608092300062584   \n',
 'TIC 320228013       6.800537835052977   \n',
 'TIC 256512446       8.057810752               \n',
 'TIC 342694097       4.350817864         \n',
 'TIC 22529346        1.274941484911037   \n',
 'TIC 468948126       2.305278439410289   \n',
 'TIC 189414606       1.2369006532739029  \n',
 'TIC 76073981        1.646116008145134   \n',
 'TIC 322900369       3.125955541296345   \n',
 'TIC 460099092       1.6091012561219522  \n',
 'TIC 192201543       16.3444904267306    \n',
 'TIC 101302325       2.1977633815453763  \n']
'''
'''
#ju中的19个
import os
path3 = "/home/life/双星偏心率的求解/汇总2021_9_30/ju_default_period.txt"
line = open(path3,'r').readlines()
new = [i[:20].rstrip(" ") for i in line]
period = [float(i[20:-1]) for i in line]

'''
new = [i[:20].rstrip(" ") for i in all]
period = [round(float(i[20:-1]),5) for i in all]
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
def GPR(x, y, yerr):
    term1 = terms.SHOTerm(sigma=1.0, rho=1.0, tau=10.0)
    term2 = terms.SHOTerm(sigma=1.0, rho=5.0, Q=0.25)
    kernel = term1 + term2
    X_new = np.linspace(-0.5, 0.5, 100)
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
            min_time_power.append(mu[i])
    bubble_sort(min_time_power)    
    min_time1 = list(mu).index(min_time_power[0]) #主极小时刻在X_new的位置
    min_time2 = list(mu).index(min_time_power[1]) #次极小时刻在X_new的位置
    return cha, min_time1, min_time2

#------------------------------------------------------------------------------------------------------------------
#  有了光变数据后循环处理光变数据，首先使用了lightkurve读取了光变数据，然后利用BLS求得光变周期，之后进行求偏心率的过程  |
#------------------------------------------------------------------------------------------------------------------
IDdir = cross_ssd_data(new)
result = open("/home/life/双星偏心率的求解/汇总2021_9_30/19ju_EEB.txt",'a')
err_ID1 = [] #存放有些打不开的fits文件
pp = period
for ii in range(len(IDdir)):
    dirall = []
    for j in IDdir[ii]:
        try:
            dirall.append(lk.TessLightCurveFile(j))
        except:
            err_ID1.append(j)
            print(j)
    lc0 = lk.LightCurveCollection(dirall)
    lc = lc0.stitch().remove_nans()
    max_period = pp[ii] 
    bin_n = 2000/len(lc.bin(1)) 
    lcc =  lc.bin(round(1/bin_n,5)).remove_nans()   
    fold = lcc.fold(max_period,normalize_phase=True,epoch_time=lcc.time.value[list(lcc.flux.value).index(min(lcc.flux.value))]-0.40*max_period)
    fold1 = lc.fold(max_period,normalize_phase=True,epoch_time=lcc.time.value[list(lcc.flux.value).index(min(lcc.flux.value))]-0.40*max_period)
    x = fold.phase.value
    y = fold.flux.value
    yerr = fold.flux_err.value  
    mu, sigma, X_new = GPR(x, y, yerr)#GPR结果
    cha, min_time1, min_time2 = difference(mu)#差分寻峰结果
    mean_cha = np.mean(cha)
    mean_sigma = np.mean(sigma) 
    X_new[min_time1] #主极小时刻在X_new的位置
    X_new[min_time2] #次极小时刻在X_new的位置
    #求掩食star和end的位置
    for i in range(min_time1,1,-1):
        h_c1 = mu[i]-mu[i-1]
        if h_c1 > 0:
            star1_p = i
            break
    for i in range(min_time2,1,-1):
        h_c2 = mu[i]-mu[i-1]
        if h_c2 > 0:
            star2_p = i
            break
    for i in range(min_time1,len(X_new)-1):
        q_c1 = mu[i]-mu[i+1]
        if q_c1 > 0:
            end1_p = i
            break
    for i in range(min_time2,len(X_new)-1):
        q_c2 = mu[i]-mu[i+1]
        if q_c2 > 0:
            end2_p = i
            break
    try:
        #得到正确的star和end在X_new的位置和在y为cha_mean-2*mean时的x值，y为cha_mean+2*mean时的x值
        true_star1 = X_new[star1_p]
        true_end1 = X_new[end1_p]
        true_star2 = X_new[star2_p]
        true_end2 = X_new[end2_p]
        #将掩食部分data截取下来
        ss1 = []
        for i in range(len(fold1.phase.value)):
            ss1.append(abs(fold1.phase.value[i]-true_star1))
        s1 = ss1.index(min(ss1))
        
        ss2 = []    
        for i in range(len(fold1.phase.value)):
            ss2.append(abs(fold1.phase.value[i]-true_star2))
        s2 = ss2.index(min(ss2))
        ee1 = []  
        for i in range(len(fold1.phase.value)):
            ee1.append(abs(fold1.phase.value[i]-true_end1))
        e1 = ee1.index(min(ee1))
        ee2 = []       
        for i in range(len(fold1.phase.value)):
            ee2.append(abs(fold1.phase.value[i]-true_end2))
        e2 = ee2.index(min(ee2))
        
        x_f = fold1.phase.value[s1:e1]
        y_f = fold1.flux.value[s1:e1]
        x1_f = fold1.phase.value[s2:e2]
        y1_f = fold1.flux.value[s2:e2]
        #开始拟合取出的数据
        yy = [i for i in y_f]
        def f(x,a,b,c,d):
            return a * np.exp(-(x-b)**2/(2.0*c**2))+d
        param_bounds = ([-1,round(X_new[min_time1]-0.05,2),0,0],[0,round(X_new[min_time1]+0.05,2),np.inf,np.inf])
        p_opt, p_cov = scipy.optimize.curve_fit(f,x_f, yy,maxfev = 10000,bounds=param_bounds)
        a,b,c,d = p_opt
        best_fit_gauss_2 = f(x_f,a,b,c,d)
        print(p_opt)
        print('Amplitude: {} +\- {}'.format(p_opt[0], np.sqrt(p_cov[0,0])))
        print('Mean: {} +\- {}'.format(p_opt[1], np.sqrt(p_cov[1,1])))
        print('Standard Deviation: {} +\- {}'.format(p_opt[2], np.sqrt(p_cov[2,2])))
        yy1 = [i for i in y1_f]
        def f(x,a,b,c,d):
            return a * np.exp(-(x-b)**2/(2.0*c**2))+d
        param_bounds1 = ([-1,round(X_new[min_time2]-0.05,2),0,0],[0,round(X_new[min_time2]+0.05,2),np.inf,np.inf])
        p_opt1, p_cov1 = scipy.optimize.curve_fit(f,x1_f, yy1,maxfev = 10000,bounds=param_bounds1)
        a,b,c,d = p_opt1
        best_fit_gauss_21 = f(x1_f,a,b,c,d)
        print(p_opt1)
        print('Amplitude: {} +\- {}'.format(p_opt1[0], np.sqrt(p_cov1[0,0])))
        print('Mean: {} +\- {}'.format(p_opt1[1], np.sqrt(p_cov1[1,1])))
        print('Standard Deviation: {} +\- {}'.format(p_opt1[2], np.sqrt(p_cov1[2,2])))
    
        eccentric = e(p_opt,p_opt1)
        result.write("{: <20}".format(lc.label))
        result.write("{: <20}".format(lc.ra))
        result.write("{: <20}".format(lc.dec))
        result.write("{: <20}".format(str(pp[ii])))
        result.write("{: <20}".format(str(eccentric)))
        result.write("\n")
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
        #pg1.plot(ax=ax4,label='',c='k').axvline(max_period,lw=1.0, ls='dashed',c='black')
        ax4.tick_params(which='both',labelsize=10,direction='in',top=True, right=True)
        ax4.text(.020, .7, '(b)',
                            horizontalalignment='left',
                            transform=ax4.transAxes, fontsize=15)
        ax2 = plt.subplot(gs[1,2:])
        plt.plot(X_new[1:], cha,c='k')
        plt.scatter(X_new[1:], cha,c='',edgecolors='k')
        #plt.axhline(0,c='gray',ls='--')
        plt.axhline(mean_cha+4*mean_sigma,c='gray',ls='-',lw=1)
        plt.axhline(mean_cha-4*mean_sigma,c='gray',ls='-',lw=1) 
        plt.axvline(true_star1,c='r',ls='--',lw=1)
        plt.axvline(true_star2,c='r',ls='--',lw=1)
        plt.axvline(true_end1+(X_new[1]-X_new[0]),c='r',ls='-',lw=1)
        plt.axvline(true_end2+(X_new[1]-X_new[0]),c='r',ls='-',lw=1)
        #plt.axvline(X_new[min_time2],c='k',ls='--',lw=1)
        #plt.axvline(X_new[min_time1],c='k',ls='--',lw=1)
        ax2.set_xlabel('Phase',fontsize=10)
        ax2.set_ylabel('Difference',fontsize=10)
        ax2.tick_params(which='both',labelsize=10,direction='in',top=True, right=True)
        ax2.text(.020, .7, '(c)',
                            horizontalalignment='left',
                            transform=ax2.transAxes, fontsize=15)
        ax1 = plt.subplot(gs[2,2:])
        #for c in sample_pred['y_pred']:
        #    plt.plot(X_new,c,"gray",alpha=0.1)
        fold1.scatter(ax=ax1,s=0.1,c='gray',label='')
        plt.scatter(X_new, mu ,c='',edgecolors='k', label="GPR");
        #plt.fill_between(X_new.flatten(), mu - 2 * sigma, mu + 2 * sigma, color="gray", alpha=0.5)
        plt.axvline(true_star1,c='r',ls='--',lw=1)
        plt.axvline(true_star2,c='r',ls='--',lw=1)
        plt.axvline(true_end1,c='r',ls='--',lw=1)
        plt.axvline(true_end2,c='r',ls='--',lw=1)
        plt.axvspan(true_star1,true_end1,facecolor='gray',alpha=0.3)
        plt.axvspan(true_star2,true_end2,facecolor='gray',alpha=0.3)
        #plt.axvline(X_new[min_time2],c='k',ls='--',lw=1)
        #plt.axvline(X_new[min_time1],c='k',ls='--',lw=1)
        #plt.scatter(x, y, label="observed data",s=1);
        #plt.title("predictive mean and 2σ interval");
        ax1.set_xlabel('Phase',fontsize=10)
        ax1.set_ylabel('Normalized flux',fontsize=10)
        ax1.tick_params(which='both',labelsize=10,direction='in',top=True, right=True)
        ax1.text(.020, .7, '(d)',
                            horizontalalignment='left',
                            transform=ax1.transAxes, fontsize=15)    
        plt.legend()    
        #plt.subplots_adjust(hspace=0, wspace=0)
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
        ax5.text(.020, .7, '(e)',
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
        ax6.text(.020, .7, '(f)',
                            horizontalalignment='left',
                            transform=ax6.transAxes, fontsize=15)
        plt.subplots_adjust(top=0.960,bottom=0.05,left=0.09,right=0.970,hspace=0.400,wspace=0.635)
        #plt.savefig("/home/life/双星偏心率的求解/e_data_picture/"+lc.label+".eps")
        plt.savefig("/home/life/双星偏心率的求解/汇总2021_9_30/"+lc.label+".png")
    except:
        print("Period is Fales")   

