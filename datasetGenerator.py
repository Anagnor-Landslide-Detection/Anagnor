
import random
import numpy as np
from random import randrange

from netCDF4 import Dataset
from netCDF4 import date2index

from torch.utils.data import Dataset as TorchDataset

import pandas as pd

import datetime
from datetime import timedelta

import cv2

def random_date(start, end):
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)
def t(matrix): # convert to cv2 format
    return np.transpose(matrix,(1,2,0))
def rev(matrix):
    return np.transpose(matrix,(2,0,1))
class DatasetGen():
    def __init__(self,path='gistemp1200_GHCNv4_ERSSTv5.nc'):
        self.img_size = 1024
        pass
        # data = Dataset('gistemp1200_GHCNv4_ERSSTv5.nc').variables['tempanomaly']

        # temp_anomaly = data.variables['tempanomaly']
    def getData(self,timeIndex,long,lat):
        pass

def splice2Dnc(nc,iter_,x_i,x_j,y_i,y_j,cx = 'latitude', cy='longitude',swap=False):
    # print(nc.variables[cx][:].min(),nc.variables[cx][:].max(),cx)
    # print(nc.variables[cy][:].min(),nc.variables[cy][:].max(),cy)
    lat = nc.variables[cx][:]
    lon = nc.variables[cy][:]
    # print("b1")

    # All indices in bounding box:
    where_j = np.where((lon >= x_i) & (lon <= x_j))[0]
    where_i = np.where((lat >= y_i) & (lat <= y_j))[0]
    # print("b2")

    # Start and end+1 indices in each dimension:
    # print(where_i,y_i,y_j,iter_.shape)
    # print(y_i,y_j)
    if(len(where_i)==0):
        i0 = 0
        i1 = 10
    else:
        i0 = where_i[0]
        i1 = where_i[-1]+1
    # print("b3")

    j0 = where_j[0]
    j1 = where_j[-1]+1
    # print("b7")
    # print(j0,j1)
    if swap:
        return iter_[j0:j1,i0:i1]
    return iter_[i0:i1,j0:j1]

def splice3Dnc(nc,iter_,x_i,x_j,y_i,y_j,z_i,z_j):
    # print(nc.variables)
    lat = nc.variables['lat'][:]
    lon = nc.variables['lon'][:]

    # All indices in bounding box:
    where_j = np.where((lon >= x_i) & (lon <= x_j))[0]
    where_i = np.where((lat >= y_i) & (lat <= y_j))[0]

    # Start and end+1 indices in each dimension:
    i0 = where_i[0]
    i1 = where_i[-1]+1

    j0 = where_j[0]
    j1 = where_j[-1]+1
    # print(j0,j1)
    return iter_[z_i:z_j,i0:i1,j0:j1]


class SurfaceTemp(DatasetGen):
    def __init__(self, path='gistemp1200_GHCNv4_ERSSTv5.nc'):
        super().__init__(path=path)
        self.data = Dataset(path)
        self.temp_anomaly = self.data.variables['tempanomaly'][:]
    def getData(self, day,month,year, long, lat):
        # print(self.data.variables['time'])
        timeindex = date2index(datetime.datetime(year, month, day),
                       self.data.variables['time'])
        matrix = splice3Dnc(self.data,self.temp_anomaly,-3,3,-3,3,timeindex-15,timeindex)
        
        img = np.array(matrix)
        img = cv2.resize(t(img), (self.img_size,self.img_size))
        return rev(img)


class Elevation(DatasetGen):
    def __init__(self, path='NLDAS.nc'):
        super().__init__(path=path)
        self.data = Dataset(path)
        self.temp_anomaly = self.data.variables['elevation'][:]
    def getData(self, day,month,year, long, lat):
        matrix = []
        # print("A")
        matrix = splice2Dnc(self.data,self.temp_anomaly,long-3,long+3,lat-3,lat+3)
        # print("B")
        
        img = np.array([matrix])
        # print(t(img).shape)
        # print(1)
        img = cv2.resize(t(img), (self.img_size,self.img_size))
        # print(2)
        img = np.expand_dims(img,2)
        return rev(img)

class IRprecipitation(DatasetGen):
    def __init__(self, path='TRMMDataset/3B42_Daily.{:04d}{:02d}{:02}.7.nc4'):
        super().__init__(path=path)
        # data = Dataset('TRMMDataset/3B42_Daily.19980101.7.nc4')
        self.path = path
        self.dt = datetime.timedelta(days=3)
        pass
        # self.data = Dataset(path)
        # self.temp_anomaly = self.data.variables['elevation'][:]
    def getData(self, day,month,year, long, lat):
        # print(p)
        # print((p+).year)
        # print(nc.variables)
        # matrix = []
        # print(vars)
        last = None
        for dt in range(-15,1):
            p_new = datetime.datetime(year,month,day)-self.dt
            year,month,day = p_new.year,p_new.month,p_new.day
            file_name = ((self.path).format(year,month,day))
            nc = Dataset(file_name)
            vars = nc.variables["IRprecipitation_cnt"][:]
            matrix = splice2Dnc(nc,vars,long-3,long+3,lat-3,lat+3,'lat','lon',swap=True)
            matrix = np.array([matrix])
            # print(matrix.shape)
            try:
                last = np.vstack((matrix,last))
            except:
                last = matrix
        # print(last.shape)
        last =last.astype(np.float32)
        img = cv2.resize(t(last), (self.img_size,self.img_size))
        return rev(img)

class CustomDataset(TorchDataset):

    def __init__(self, csv_file="nasa_global_landslide_catalog_point.csv"): 
        dataSet = pd.read_csv(csv_file)
        # print((dataSet.columns))
        self.event_date = dataSet["event_date"]
        self.lat = dataSet["latitude"]
        self.long = dataSet["longitude"]
        self.st = SurfaceTemp()
        self.el = Elevation()
        self.pres = IRprecipitation()

        self.st1 = datetime.datetime.strptime('1/1/2001 1:30 PM', '%m/%d/%Y %I:%M %p')
        self.st2 = datetime.datetime.strptime('1/1/2019 4:50 AM', '%m/%d/%Y %I:%M %p')
    def __len__(self):
        return len(self.event_date)

    def __getitem__(self, idx):
        # print(idx)
        if(idx%2==0):
            date = self.event_date[idx]
            lat = self.lat[idx]
            long = self.long[idx]
            date = datetime.datetime.fromisoformat(date)
            label = 1.0
        else:
            date = random_date(self.st1,self.st2)
            lat = random.randint(-90,90)
            long = random.randint(0,180)
            label = 0.0
        # date = date.replace(day=15)
        # lat,long = float(lat),float(long)
        # print(date,lat,long)
        if(long<0):
            long+=180
        # print(date,lat,long)
        d1 = self.st.getData(15,date.month,date.year,int(lat),int(long))
        # print("d1")
        d2 = self.el.getData(date.day,date.month,date.year,long,lat)
        # print("d2")
        d3 = self.pres.getData(date.day,date.month,2000,long,lat-50)
        # print("d3")
        # 
        return (np.vstack((d1,d2,d3)).shape),label

import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Hello")
    st = SurfaceTemp()
    el = Elevation()
    pres = IRprecipitation()
    # # day,month,year, long, lat
    d1 = st.getData(15,1,1999,1,1)
    d2 = el.getData(15,1,1999,1,1)
    d3 = pres.getData(15,1,1999,1,1)
    # print(d1.shape,d2.shape,d3.shape)
    # print(np.vstack((d1,d2,d3)).shape)
    
    x = CustomDataset()
    print(len(x))
    for a in x:
        print(a)
        print("done")
        break
    # x = Elevation()
    # day,month,year, long, lat
    # plt.imshow(i)
    # plt.show()