import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import matplotlib.mlab as mlab
from matplotlib import cm
#plot of data

def set_plot_params():
    pass
    """
    Sets the matplotlib parameters

    Returns
    -------
    None


    plt.style.use('')
    plt.rcParams.update({
        '': '',
        '': ''
    })

    """


def plot_estimates( lc ,eccentric ,X_new ,fold1 ,cha ,true_star1 ,true_star2 ,true_end1 ,true_end2 ,mu ,x_f ,x1_f ,y_f ,y1_f ,best_fit_gauss_2 ,best_fit_gauss_21 ,p_opt1 ,p_opt):
    """
    Creates a plot summarizing the results of find eclipsing

    Parameters
    ----------
    star : target.Target
        the ROSS pipeline object

    Returns
    -------
    None

    """
    target = 'heihei'
    minimum = 'heihei'

    fig = plt.figure(figsize=(13,11))
    gs = gridspec.GridSpec(4,6)
    ax3 = plt.subplot(gs[0,:])
    lc.scatter(ax=ax3,label='',c='k',s=1)
    ax3.set_title(lc.label+"e={}".format(str(eccentric)),fontsize=15)
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

def plot_test(n):
    print(n)
    print("That's OK")
