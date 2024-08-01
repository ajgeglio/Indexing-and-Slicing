# https://stackoverflow.com/questions/2922532/obtain-latitude-and-longitude-from-a-geotiff-file
# get the existing coordinate system
import glob
import pandas as pd
import os
from osgeo import gdal, osr
import numpy as np

class get_coordinates:
    def __init__(self) -> None:
        pass

    def reef_overlap(df_tiffs, df_collects):
        df = df_collects.copy()
        apx = (1/60)/2
        out_df = pd.DataFrame(columns=["COLLECT ID", "OP_DATE", "SITE NAME", "SURVEY NAME", "LATTITUDE", "LONGITUDE", "REEF"])
        for i in range(len(df_tiffs)):
            reef = df_tiffs.loc[i]
            idx = df[(reef.max_lat > df.LATTITUDE - apx) & (reef.min_lat < df.LATTITUDE + apx) & (reef.max_lon > df.LONGITUDE - apx) & (reef.min_lon < df.LONGITUDE + apx)].index
            new_df = df.loc[idx]
            new_df['REEF'] = reef.reef
            out_df = pd.concat([out_df, new_df])
        return out_df

    def get_min_max_xy(self, tif_file):
        ds = gdal.Open(tif_file)
        old_cs= osr.SpatialReference()
        old_cs.ImportFromWkt(ds.GetProjectionRef())
        # create the new coordinate system
        wgs84_wkt = """GEOGCS["WGS 84", DATUM["WGS_1984", SPHEROID["WGS 84", 6378137, 298.257223563, 
                        AUTHORITY["EPSG","7030"]], AUTHORITY["EPSG","6326"]], PRIMEM["Greenwich",0, 
                        AUTHORITY["EPSG","8901"]], UNIT["degree",0.01745329251994328, 
                        AUTHORITY["EPSG","9122"]], AUTHORITY["EPSG","4326"]]"""
        new_cs = osr.SpatialReference()
        new_cs .ImportFromWkt(wgs84_wkt)
        # create a transform object to convert between coordinate systems
        transform = osr.CoordinateTransformation(old_cs,new_cs) 
        #get the point to transform, pixel (0,0) in this case
        width = ds.RasterXSize
        height = ds.RasterYSize
        gt = ds.GetGeoTransform()
        miny = gt[0]
        minx = gt[3] + width*gt[4] + height*gt[5] 
        maxy = gt[0] + width*gt[1] + height*gt[2] 
        maxx = gt[3]
        #get the coordinates in lat long
        return transform.TransformPoint(miny,minx)[0:2], transform.TransformPoint(maxy,maxx)[0:2]

    def return_min_max_tif_df(self, tif_files=None):
        min_max = [self.get_min_max_xy(t) for t in tif_files]
        names = [os.path.basename(t) for t in tif_files]
        # reefs = [os.path.basename(os.path.dirname(t)) for t in tif_files]
        reefs = [n.split("_")[0] for n in names]
        min_y, min_x = [min_max[i][0][0] for i in range(len(min_max))], [min_max[i][0][1] for i in range(len(min_max))]
        max_y, max_x  = [min_max[i][1][0] for i in range(len(min_max))], [min_max[i][1][1] for i in range(len(min_max))]
        min_max_df = pd.DataFrame(np.c_[names, min_y, min_x, max_y, max_x], columns=["filename", "min_lat", "min_lon", "max_lat", "max_lon"])
        min_max_df.min_lat = min_max_df.min_lat.astype('float')
        min_max_df.min_lon = min_max_df.min_lon.astype('float')
        min_max_df.max_lat = min_max_df.max_lat.astype('float')
        min_max_df.max_lon = min_max_df.max_lon.astype('float')
        min_max_df["avr_lat"] = (min_max_df.min_lat + min_max_df.max_lat)/2
        min_max_df["avr_lon"] = (min_max_df.min_lon + min_max_df.max_lon)/2
        min_max_df["reef"] = reefs
        return min_max_df
    

    def return_headers_min_max_coord(self, df):
        collects = df.collect_id.unique()
        coords = []
        for c in collects:
            df_tmp = df[df.collect_id == c]
            lats = df_tmp.Lat_DD.dropna()
            lons = df_tmp.Long_DD.dropna()
            # list1 = [lats.min(), lons.min(), lats.max(), lons.max()]
            list = [np.percentile(lats, 10), np.percentile(lons, 10), np.percentile(lats, 90), np.percentile(lons, 90), np.percentile(lats, 50), np.percentile(lons, 50)]
            coords.append(list)
        collects_lat_lon = pd.DataFrame(np.c_[collects, coords], columns=["collect_id", "min_lat", "min_lon", "max_lat", "max_lon", "med_lat", "med_lon"])
        return collects_lat_lon
    

    def return_MissionLog_min_max_coord(self, collect):
        log_paths = glob.glob(os.path.join(collect,'logs','*','Logs','*.log'))
        try:
            dfs = [pd.read_csv(file, header=0, delimiter=';').dropna(axis=1) for file in log_paths]
            log_df = pd.concat(dfs, axis=0, ignore_index=True)
            lats = log_df['Latitude'].values
            lons = log_df['Longitude'].values
            lats = lats[(lats<50) & (lats>41)] # Sometimes there are erroneous lat lon values in the .log files
            lons = lons[(lons>-92.5) & (lons<-75.5)]
            min_lat, min_lon = lats.min(), lons.min()
            max_lat, max_lon = lats.max(), lons.max()
        except:
            min_lat, min_lon, max_lat, max_lon = np.nan, np.nan, np.nan, np.nan
        return min_lat, min_lon, max_lat, max_lon
    
    def return_MissionLog_min_max_df(self, collect_list):
        items = []
        for collect in collect_list:
            min_lat, min_lon, max_lat, max_lon = self.return_MissionLog_min_max_coord(collect)
            item = [collect, min_lat, min_lon, max_lat, max_lon]
            items.append(item)
        df = pd.DataFrame(items, columns=["collect_path", "min_lat", "min_lon", "max_lat", "max_lon"])
        cid = lambda x: os.path.basename(x)
        d = lambda x: x.split('_')[0]
        cn = lambda x: x.split('_')[1]
        iv = lambda x: x.split('_')[2]
        cs = lambda x: x.split('_')[3]
        df["collect_id"] = df.collect_path.map(cid)
        df["date"] = df.collect_id.apply(d)
        df["date"] = pd.to_datetime(df.date, format="%Y%m%d")
        df["collect"] = df.collect_id.apply(cn)
        df["AUV"] = df.collect_id.apply(iv)
        df["cam_sys"] = df.collect_id.apply(cs)
        df = df[["collect_path", "collect_id", "date", "collect", "AUV", "cam_sys", "min_lat", "min_lon", "max_lat", "max_lon"]]
        return df
    
if __name__ == '__main__':
    get_coordinates()