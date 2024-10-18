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

    def reef_overlap(df_tiffs, OP_TABLE_path):
        OP_TABLE_df = pd.read_excel(OP_TABLE_path)
        OP_TABLE_df = OP_TABLE_df[["COLLECT ID", "OP_DATE", "SITE NAME", "SURVEY NAME", "LATITUDE", "LONGITUDE"]]
        df = OP_TABLE_df.copy()
        apx = (1/60)/2
        out_df = pd.DataFrame(columns=["COLLECT ID", "OP_DATE", "SITE NAME", "SURVEY NAME", "LATITUDE", "LONGITUDE", "REEF"])
        for i in range(len(df_tiffs)):
            reef = df_tiffs.loc[i]
            idx = df[(reef.max_lat > df.LATITUDE - apx) & (reef.min_lat < df.LATITUDE + apx) & (reef.max_lon > df.LONGITUDE - apx) & (reef.min_lon < df.LONGITUDE + apx)].index
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

    def return_min_max_tif_df(self, reef_dict, tif_files=None):
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
        min_max_df["filepath"] = tif_files
        min_max_df = min_max_df.drop_duplicates(subset="reef").reset_index(drop=True)
        min_max_df["Reef_Name"] = min_max_df.reef.apply(lambda x: reef_dict[x])
        return min_max_df
    
    def log_col_lat_lon(self, root):
        out_df = pd.DataFrame()
        for collect_id in os.listdir(root):
            logpath = r"logs\*\Logs\*.log"
            folder = os.path.join(root,collect_id,logpath)
            logs = glob.glob(folder)
            log_df = pd.DataFrame()
            for log in logs:
                log_df_t = pd.read_csv(log, delimiter=";")
                log_df = pd.concat([log_df, log_df_t])
            try:
                med_lat, med_lon = log_df.Latitude.median(), log_df.Longitude.median()
            except:
                med_lat, med_lon = np.nan, np.nan
            df = pd.DataFrame([[collect_id, med_lat, med_lon]], columns=["collect_id", "Latitude", "Longitude"])
            out_df = pd.concat([out_df, df])
        return out_df
    
    def return_headers_min_max_coord(self, df):
        collects = df.collect_id.unique()
        coords = []
        for c in collects:
            df_tmp = df[df.collect_id == c]
            df_tmp = df_tmp[df_tmp.Usability == "Usable"]
            lats = df_tmp.Latitude.dropna()
            lons = df_tmp.Longitude.dropna()
            try:
                # list1 = [lats.min(), lons.min(), lats.max(), lons.max()]
                list = [np.percentile(lats, 10), np.percentile(lons, 10), np.percentile(lats, 90), np.percentile(lons, 90), np.percentile(lats, 50), np.percentile(lons, 50)]
            except IndexError:
                print("not lat lon in ", c)
                list = [np.nan]*6
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