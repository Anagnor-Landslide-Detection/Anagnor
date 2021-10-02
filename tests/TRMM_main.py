import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
data = Dataset('TRMMDataset/3B42_Daily.19980101.7.nc4')

from netCDF4 import date2index
# from datetime import datetime
# timeindex = date2index(datetime(2019, 1, 15),
#                        data.variables['time'])
# print(data.variables)
lat = data.variables['lon'][:]
lat = data.variables['lat'][:]
# print(lat[0]-lat[1])
# print(lat[1]-lat[2])
# print(lat)
# quit()
# lon = data.variables['lon'][:]
# lon, lat = np.meshgrid(lon, lat)
# print(data.variables)
temp_anomaly = data.variables['HQprecipitation']
# print(temp_anomaly)

# print(temp_anomaly[0,-179.1])
plt.imshow(temp_anomaly)
plt.show()
# fig = plt.figure(figsize=(10, 8))
# m = Basemap(projection='lcc', resolution='c',
#             width=8E6, height=8E6, 
#             lat_0=45, lon_0=-100,)
# m.shadedrelief(scale=0.5)
# m.pcolormesh(lon, lat, temp_anomaly,
#              latlon=True, cmap='RdBu_r')
# plt.clim(-8, 8)
# m.drawcoastlines(color='lightgray')

# plt.title('January 2014 Temperature Anomaly')
# plt.colorbar(label='temperature anomaly (°C)');