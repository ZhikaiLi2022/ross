import numpy as np
from scipy.optimize import curve_fit
import celerite2
from celerite2 import terms

from .lightcurve import Target
from . import plots
#example
# >>> import ross
# >>> import matplotlib.pyplot as plt
# >>> tic = 'TIC 401606923'
# >>> period = 13.91854
# >>> tar = ross.Analysis(tic).cross_ssd_data().fits_open().fold(period).GPR(200).difference().remove_peak(lim1=None,sub=1).remove_peak(lim1=None,sub=2).GSfit(sub=1).GSfit(sub=2)
#----------------
#  Bubble sorting|
#----------------
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
#----------------------------------------------
#  compare with e calculated through formula  |
#----------------------------------------------
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

class Analysis(Target):
    #----------------------
    # Define GPR function |
    #----------------------
    def GPR(self,X_n):
        term1 = terms.SHOTerm(sigma=1.0, rho=1.0, tau=10.0)
        term2 = terms.SHOTerm(sigma=1.0, rho=5.0, Q=0.25)
        kernel = term1 + term2
        self.X_new = np.linspace(-0.5, 0.5, X_n)
        gp = celerite2.GaussianProcess(kernel, mean=0.0)
        gp.compute(self.fold_lc_bin.phase.value,yerr = self.fold_lc_bin.flux_err.value)
        print("Initial log likelihood: {0}".format(gp.log_likelihood(self.fold_lc_bin.flux.value)))
        self.mu, self.variance = gp.predict(self.fold_lc_bin.flux.value, self.X_new, return_var=True)
        self.sigma = np.sqrt(self.variance)
        return self #mu, sigma, X_new   
    
    #----------------------------
    #  Differential calculation |
    #---------------------------- 
    def difference(self):
        self.cha = [self.mu[i]-self.mu[i-1] for i in range(1,len(self.mu))]
        min_time_power = []
        for i in range(1,len(self.cha)):
            if self.cha[i]/abs(self.cha[i]) == -self.cha[i-1]/abs(self.cha[i-1]):                
                if self.cha[i] > 0:
                    min_time_power.append(self.mu[i])
        bubble_sort(min_time_power) 
        min_time= [] 
        for i in min_time_power:  #if i < 1-(1-min_time_power[0])*0.05:
            min_time.append(list(self.mu).index(i))    
            self.min_time = min_time
        return self #cha min_time 
            
    #---------------
    # Extract peak |      
    #---------------
    '''
    lim1的作用是在一定范围(0.05)内寻峰,如果=None就不限定范围。
    sub是第一个峰min_time1在min_time的序列位置，去除这个峰选取另一个作为第二个峰
    '''
    def remove_peak(self ,lim1 ,sub):   
        X_new    = self.X_new
        cha      = self.cha
        mu       = self.mu
        min_time = self.min_time
        fold1    = self.fold_lc
        if lim1 !=None:   #差0.5相位处左右0.05flux最低的。 
            result = [abs(lim1-X_new[i]) for i in min_time]
            result1 = []
            for i in result:
                if i < 0.05:
                    result1.append(i)
            min_time1 = min_time[result.index(min(result1))]
        else: min_time1 = min_time[0]
        if sub != None:
            if sub == 1:
                min_time1 = min_time[0]
            if sub ==2:
                min_time1 = min_time[1]
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
        if sub != None:
            if sub == 1:
                self.x_f        = x_f
                self.y_f        = y_f
                self.true_star1 = true_star1
                self.true_end1  = true_end1
                self.min_time1  = min_time1
                return self #x_f, y_f, true_star1, true_end1, min_time1
            if sub == 2:
                self.x1_f        = x_f
                self.y1_f        = y_f
                self.true_star2 = true_star1
                self.true_end2  = true_end1
                self.min_time2  = min_time1
                return self #x1_f, y1_f, true_star2, true_end2, min_time2

    
    #-------------------------
    # Fit the extracted data |
    #-------------------------
    '''
        print(p_opt)
        print('Amplitude: {} +\- {}'.format(p_opt[0], np.sqrt(p_cov[0,0])))
        print('Mean: {} +\- {}'.format(p_opt[1], np.sqrt(p_cov[1,1])))
        print('Standard Deviation: {} +\- {}'.format(p_opt[2], np.sqrt(p_cov[2,2])))
    '''
    def GSfit(self,sub): #x_f,y_f,X_new,min_time1 
        if sub != None:
            if sub == 1:
                x_f       = self.x_f
                y_f       = self.y_f
                X_new     = self.X_new
                min_time1 = self.min_time1
            if sub == 2:
                x_f       = self.x1_f
                y_f       = self.y1_f
                X_new     = self.X_new
                min_time1 = self.min_time2
        yy = [i for i in y_f]
        def f(x,a,b,c,d):
            return a * np.exp(-(x-b)**2/(2.0*c**2))+d
        param_bounds = ([-1,round(X_new[min_time1]-0.05,2),0,0],[0,round(X_new[min_time1]+0.05,2),np.inf,np.inf])
        p_opt, p_cov = curve_fit(f,x_f, yy,maxfev = 10000,bounds=param_bounds)
        a,b,c,d = p_opt
        best_fit_gauss_2 = f(x_f,a,b,c,d)
        if sub != None:
            if sub == 1:
                self.p_opt            = p_opt 
                self.best_fit_gauss_2 = best_fit_gauss_2
                return self #p_opt, best_fit_gauss_2
            if sub == 2:
                self.p_opt1            = p_opt 
                self.best_fit_gauss_21 = best_fit_gauss_2
                return self #p_opt1, best_fit_gauss_21
    def pipeline(self,period):
        period = period
        self.cross_ssd_data().fits_open().fold(period).GPR(200).difference().remove_peak(lim1=None,sub=1).remove_peak(lim1=None,sub=2).GSfit(sub=1).GSfit(sub=2)
        return self
    def plot_results(self):
        lc                = self.lc                  
        eccentric         = 0           
        X_new             = self.X_new               
        cha               = self.cha                 
        true_star1        = self.true_star1          
        true_star2        = self.true_star2          
        true_end1         = self.true_end1           
        true_end2         = self.true_end2           
        mu                = self.mu                  
        x_f               = self.x_f                 
        x1_f              = self.x1_f                
        y_f               = self.y_f                 
        y1_f              = self.y1_f                
        best_fit_gauss_2  = self.best_fit_gauss_2    
        best_fit_gauss_21 = self.best_fit_gauss_21   
        p_opt             = self.p_opt               
        p_opt1            = self.p_opt1              
        fold1             = self.fold_lc
        plots.plot_estimates( lc ,eccentric ,X_new ,fold1 ,cha ,true_star1 ,true_star2 ,true_end1 ,true_end2 ,mu ,x_f ,x1_f ,y_f ,y1_f ,best_fit_gauss_2 ,best_fit_gauss_21 ,p_opt1 ,p_opt)
    def pp(self):
        n = self.p_opt
        plots.plot_test(n)
