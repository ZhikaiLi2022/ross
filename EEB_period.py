import os
path3 = "/home/life/双星偏心率的求解/汇总2021_9_30/在ju中但未被我们找到的/"
line3 = [i[1:-4] for i in os.listdir(path3)]
len(new) #19

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
IDdir = cross_ssd_data(new)
#-----------------------------------------------------------------------------------------------------------------------------
#  有了光变数据后循环处理光变数据，首先使用了lightkurve读取了光变数据，然后利用LS求光变周期保存，之后再在另一个code进行分类的过程  |
#-----------------------------------------------------------------------------------------------------------------------------
path3 = '/home/life/双星偏心率的求解/汇总2021_9_30/ju_default_period.txt'
f = open(path3,'a')
err_ID1 = [] #存放有些打不开的fits文件
period = []  #存放初始周期。
for ii in range(len(IDdir)):
    dirall = []
    for j in IDdir[ii]:
        try:
            dirall.append(lk.TessLightCurveFile(j))
        except:
            err_ID1.append(j)
            print(j)
    sector = []
    dirall1 = [dirall[0]]
    for i in dirall:
        sector.append(int(i.sector))
    if len(dirall) > 1: 
        for i in range(1,len(sector)):
            if sector[i]-sector[i-1] == 1:
                dirall1 = dirall[i-1:i+1]
                break
    lc0 = lk.LightCurveCollection(dirall1)
    lc = lc0.stitch().remove_nans()
    pg = lc.to_periodogram( method="BLS",minimum_period=1,maximum_period=8,frequency_factor=1.3)
    #pg = lc.to_periodogram( method="LS",minimum_period=1,maximum_period=8,oversample_factor=4000)  
    period.append(pg.period_at_max_power)
    lc.fold(pg.period_at_max_power).scatter()
    fold1 = lc.remove_nans().fold(pg.period_at_max_power.value)
    fold3 = lc.remove_nans().fold(pg.period_at_max_power.value*2)
    fold4 = lc.remove_nans().fold(pg.period_at_max_power.value*4)
    nxp = [i*pg.period_at_max_power.value for i in range(1,9)]
    print(nxp)
    fig = plt.figure(figsize=(12,13))
    ax0 = plt.subplot(5,2,1)
    lc.plot(ax=ax0)
    plt.legend(fontsize=12,loc='upper right')
    ax1 = plt.subplot(5,2,2)
    pg.plot(ax=ax1,label=str(round(pg.period_at_max_power.value,4)),view='period').axvline(pg.period_at_max_power.value,lw=1.0, ls='dashed',c='black')
    plt.legend(fontsize=12,loc='upper right')
    ax2 = plt.subplot(5,1,2)
    pg.plot(ax=ax2,label="narrow pg",view='period').axvline(pg.period_at_max_power.value,lw=1.0, ls='dashed',c='black')
    plt.legend(fontsize=12,loc='upper right')
    ax3 = plt.subplot(5,1,3)
    fold1.scatter(ax=ax3,label='P') 
    plt.legend(fontsize=12,loc='upper right')   
    ax4 = plt.subplot(5,1,4)
    fold3.scatter(ax=ax4,label='P*2')
    plt.legend(fontsize=12,loc='upper right')
    ax5 = plt.subplot(5,1,5)
    fold4.scatter(ax=ax5,label='P*4')
    plt.legend(fontsize=12,loc='upper right')
    plt.subplots_adjust(top=0.980,bottom=0.05,left=0.09,right=0.970,hspace=0.25, wspace=0.25)
    plt.show()
    period_true = pg.period_at_max_power.value
    for k in range(100):
        r = input("周期是否已经选好？")
        if r == "ok":   
            break
        else:
            period_true = float(input("请给定一个正确的周期："))
            fold1 = lc.remove_nans().fold(period_true)
            fold3 = lc.remove_nans().fold(period_true+0.01)
            fold4 = lc.remove_nans().fold(period_true-0.01)
            fig = plt.figure(figsize=(12,13))
            ax0 = plt.subplot(5,2,1)
            lc.plot(ax=ax0)
            plt.legend(fontsize=12,loc='upper right')
            ax1 = plt.subplot(5,2,2)
            pg.plot(ax=ax1,label=str(round(pg.period_at_max_power.value,4)),view='period').axvline(pg.period_at_max_power.value,lw=1.0, ls='dashed',c='black')
            plt.legend(fontsize=12,loc='upper right')
            ax2 = plt.subplot(5,1,2)
            pg.plot(ax=ax2,label="narrow pg",view='period').axvline(pg.period_at_max_power.value,lw=1.0, ls='dashed',c='black')
            plt.legend(fontsize=12,loc='upper right')
            ax3 = plt.subplot(5,1,3)
            fold1.scatter(ax=ax3,label='P') 
            plt.legend(fontsize=12,loc='upper right')   
            ax4 = plt.subplot(5,1,4)
            fold3.scatter(ax=ax4,label='P*2')
            plt.legend(fontsize=12,loc='upper right')
            ax5 = plt.subplot(5,1,5)
            fold4.scatter(ax=ax5,label='P*4')
            plt.legend(fontsize=12,loc='upper right')
            plt.subplots_adjust(top=0.980,bottom=0.05,left=0.09,right=0.970,hspace=0.25, wspace=0.25)
            plt.show()

    f.write("{: <20}".format(str(lc.label)))
    f.write("{: <20}".format(str(period_true)))
    f.write("\n")
    fig = plt.figure(figsize=(12,13))
    ax0 = plt.subplot(5,2,1)
    lc.plot(ax=ax0)
    plt.legend(fontsize=12,loc='upper right')
    ax1 = plt.subplot(5,2,2)
    pg.plot(ax=ax1,label=str(round(pg.period_at_max_power.value,4)),view='period').axvline(pg.period_at_max_power.value,lw=1.0, ls='dashed',c='black')
    plt.legend(fontsize=12,loc='upper right')
    ax2 = plt.subplot(5,1,2)
    pg.plot(ax=ax2,label="narrow pg",view='period').axvline(pg.period_at_max_power.value,lw=1.0, ls='dashed',c='black')
    plt.legend(fontsize=12,loc='upper right')
    ax3 = plt.subplot(5,1,3)
    fold1.scatter(ax=ax3,label=str(round(period_true,6))) 
    plt.legend(fontsize=12,loc='upper right')   
    ax4 = plt.subplot(5,1,4)
    fold3.scatter(ax=ax4,label='P*2')
    plt.legend(fontsize=12,loc='upper right')
    ax5 = plt.subplot(5,1,5)
    fold4.scatter(ax=ax5,label='P*4')
    plt.legend(fontsize=12,loc='upper right')
    plt.subplots_adjust(top=0.980,bottom=0.05,left=0.09,right=0.970,hspace=0.25, wspace=0.25)
    plt.savefig("/home/life/双星偏心率的求解/汇总2021_9_30/juEEBperiod/"+lc.label+".png")
    plt.close('all')
f.close()

