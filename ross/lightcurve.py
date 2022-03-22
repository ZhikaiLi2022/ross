import os

import ross
from lightkurve import TessLightCurveFile
from lightkurve import LightCurveCollection

#----------------------------
#  在硬盘上匹配光变曲线数据  |
#----------------------------
class Target:
    def __init__(self,id):
        self.id = id
    def cross_ssd_data(self): #ID的格式为"TIC 224292441"
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
        for i in range(len(file_dir)):
            for j in range(len(file_dir[i])):
                a = self.id[4:]
                if a == file_dir[i][j][-len(a)-15:-15] and file_dir[i][j][-len(a)-16:-len(a)-15] == '0':
                    IDdir.append(file_dir[i][j])
                    continue
        print("IDdir--"+str(i))
        self.iddir = IDdir
        return self
    def fits_open(self):
        """
        Read the file according to the .fits of target.
    
        Parameters
        ----------
        iddir : list
    
        Returns
        -------
        lc : Light Curve Object from lightkurve
    
        """
        fits_err = []
        dirall = []
        for i in self.iddir:
            try:
                dirall.append(TessLightCurveFile(i))
            except:
                fits_err.append(i)
        if dirall == []:
            return print("The all of .fits filr of target is False")
        if len(dirall) > 5:
            dirall = dirall[:5]
        coll = LightCurveCollection(dirall)
        self.lc = coll.stitch().remove_nans()
        return self
    def fold(self,initial_period):
        self.period = initial_period
        bin_n = (2000)/len(self.lc.bin(1)) 
        lc_bin =  self.lc.bin(round(1/bin_n,5)).remove_nans()   
        fold = lc_bin.fold(initial_period,normalize_phase=True,epoch_time=lc_bin.time.value[list(lc_bin.flux.value).index(min(lc_bin.flux.value))]-0.40*initial_period)
        fold1 = self.lc.fold(initial_period,normalize_phase=True,epoch_time=lc_bin.time.value[list(lc_bin.flux.value).index(min(lc_bin.flux.value))]-0.40*initial_period)
        self.fold_lc_bin = fold
        self.fold_lc = fold1
        return self
