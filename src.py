import re
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import glob
import pickle

class return_time:
    def __init__(self) -> None:
        pass
    def get_time_obj(self, time_s):
        # return datetime.datetime.fromtimestamp(time_s)
        return datetime.datetime.fromtimestamp(time_s) if pd.notnull(time_s) else np.nan
    def get_Y(self, time_s):
        return self.get_time_obj(time_s).strftime('%Y') if pd.notnull(time_s) else np.nan
    def get_m(self, time_s):
        return self.get_time_obj(time_s).strftime('%m') if pd.notnull(time_s) else np.nan
    def get_d(self, time_s):
        return self.get_time_obj(time_s).strftime('%d') if pd.notnull(time_s) else np.nan
    def get_t(self, time_s):
        return self.get_time_obj(time_s).strftime('%H:%M:%S') if pd.notnull(time_s) else np.nan

def bagfile_from_image(filename, aluim_path, window=30):
    image_id = filename.split(".")[0]
    time_sec = float(image_id.split("_")[1] +"." + image_id.split("_")[2])
    print("Image Date and Time:")
    print(datetime.datetime.fromtimestamp(time_sec).strftime('%Y-%m-%d %H:%M:%S'))
    aluim = pd.read_csv(aluim_path, index_col=0, low_memory=False)
    aluim21_window = aluim[(aluim.Time_s >= time_sec-window/2) & (aluim.Time_s <= time_sec+window/2)]
    print(aluim21_window.BagFile.values)

def copy_img_df(df, dest_folder):
    l = len(df)
    i=0
    for f, n in zip(df.image_path, df.filename):
        src = f
        dst = os.path.join(dest_folder, n)
        # File copy was interrupted often due to network, added src/dest comparison
        if os.path.exists(src):
            if os.path.exists(dst):
                if os.stat(src).st_size == os.stat(dst).st_size:
                    i+=1
                else:
                    shutil.copy(src, dst)
                    i+=1
            else:
                shutil.copy(src, dst)
                i+=1
            print("Copying", i,"/",l, end='  \r')
        else: print(f"{src} not found")

def copy_imgs(img_pths, dest_folder):
    l = len(img_pths)
    for i, img_pth in enumerate(img_pths):
        src, dst = img_pth, dest_folder
        # File copy is interrupted often due to network, added src/dest comparison
        if os.path.exists(src):
            if os.path.exists(dst):
                if os.stat(src).st_size == os.stat(dst).st_size:
                    continue
                else:
                    shutil.copy(src, dst)
            else:
                shutil.copy(src, dst)
            print("Copying", i,"/",l, end='  \r')
        else: print(f"{src} not found")

def load_pickle(pickle_pth): #unpickling
    with open(pickle_pth, "rb") as fp:   
        pf = pickle.load(fp)
    return pf
def dump_pickle(pickle_pth, pf): #pickling
    with open(pickle_pth, "wb") as fp:
        pickle.dump(pf, fp)

def create_collect_id_df(imgs_glob):
    ''' create a dataframe of collect ids based on a list of images'''
    df = pd.DataFrame()
    df['image_path'] = imgs_glob
    df['filename'] = df.image_path.apply(lambda x: os.path.basename(x))
    folder = df.image_path.apply(lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))))
    pattern = r"([0-9]{8}_[0-9]{3}_[a-z,A-Z]{4}[0-9]{4}_[a-z,A-Z]{3}[0-2]{1})"
    collect_id_match = folder.apply(lambda x: re.search(pattern, x))
    collect_ids = [match.group() for match in collect_id_match if match != None]
    col_idx = collect_id_match[collect_id_match.values!=None].index
    df = df.loc[col_idx]
    df['collect_id'] = collect_ids
    return df
    
def list_files_exclude_pattern(filepath, filetype, pat):
   paths = []
   for root, dirs, files in os.walk(filepath):
      if re.search(pat, root):
         pass
      else:
         for file in files:
            if file.lower().endswith(filetype.lower()):
               paths.append(os.path.join(root, file))
   return(paths)

def list_folders_w_pattern(pth, pat):
    folders = []
    for root, dirs, files in os.walk(pth):
        for name in dirs:
            folders.append(os.path.join(root,name))
    extra_folders = [re.findall(pat,folder) for folder in folders]
    folders = [sublist for list in extra_folders for sublist in list]
    return folders

def list_collects(filepath):
   paths = []
   for root, dirs, files in os.walk(filepath):
      for dir in dirs:
        paths.append(os.path.join(root, dir))
        pat1 = '([0-9]{8}_[0-9]{3}_[a-z,A-Z]+[0-9]*_[a-z,A-Z]*[0-9]*[a-z,A-Z]*)'
        pat2 = '.*\d+_\d+_\w+_\w{4}\Z'
   collects = [re.findall(pat1, i) for i in paths]
   collects = list(set([item for sublist in collects for item in sublist]))
   return(collects)

def list_files(filepath, filetype):
   paths = []
   for root, dirs, files in os.walk(filepath):
      for file in files:
         if file.lower().endswith(filetype.lower()):
            paths.append(os.path.join(root, file))
   return(paths)

def create_empty_txt_files(filename_list, ext=".txt"):
    for fil in filename_list:
        file = fil + f"{ext}"
        with open(file, 'w'):
            continue

def make_move_df(orig_pth, new_pth, ext = ".txt"):
    descr = os.path.join(orig_pth,"*"+ext)
    files = glob.glob(descr)
    df = pd.DataFrame(files, columns=["original_path"])
    bn = lambda x: os.path.basename(x)
    df['original_filename'] = df.original_path.apply(bn)
    np = lambda x: os.path.join(new_pth,x)
    df['new_path'] = df.original_filename.apply(np)
    return df

def move_lbl_files(move_df):
    i = 0
    k = 0
    l = len(move_df)
    for src, dst in zip(move_df.original_path, move_df.new_path):
        if not os.path.exists(dst):
            shutil.copy(src,dst)
            k+=1
        i+=1
        print("file", i,"/",l,"new items found", k, end=" \r")

def make_datetime_folder():
    t = datetime.datetime.now()
    timestring = f"{t.year:02d}{t.month:02d}{t.day:02d}-{t.hour:02d}{t.minute:02d}{t.second:02d}"
    Ymmdd = timestring.split("-")[0]
    out_folder = f"2019-2023-{Ymmdd}"
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    print(out_folder)
    return out_folder, Ymmdd

def create_image_df_save_csv(image_list, out_folder, year):
    df_imgs = pd.DataFrame(image_list, columns=["image_path"])
    im = lambda x: os.path.basename(x)
    df_imgs["filename"] = df_imgs.image_path.map(im)
    # df_imgs = df_imgs.drop_duplicates(subset="image_path")
    pat1 = r'([0-9]{8}_[0-9]{3}_[a-z,A-Z]+[0-9]*_[a-z,A-Z]*[0-9]*[a-z,A-Z]*)'
    pat2 = r'([0-9]{8}_[0-9]{3}_[a-z,A-Z]+[0-9][0-9][0-9][0-9]_[a-z,A-Z]+[0-2])'
    df_imgs['collect_id'] = df_imgs["image_path"].str.extract(pat1)
    df_imgs.to_csv(os.path.join(out_folder,f"{year}_imgs.csv"))
    print(year, df_imgs.shape) # (1993212, 2) (2009213, 2) (2253118, 2)
    return df_imgs

def combine_headers(headers_paths):
    l = len(headers_paths)
    df = pd.DataFrame()
    # Concatenation loop
    # for i in range(l):
    for path in headers_paths:
        # path = headers_paths[i]
        tempdf = pd.read_csv(path, low_memory=False).drop(columns="collect_id", errors="ignore")
        tempdf.rename(columns={tempdf.columns[0]: "Time_s"}, inplace=True)
        pat1 = r'([0-9]{8}_[0-9]{3}_[a-z,A-Z]+[0-9]*_[a-z,A-Z]*[0-9]*[a-z,A-Z]*)'
        pat2 = r'([0-9]{8}_[0-9]{3}_[a-z,A-Z]+[0-9][0-9][0-9][0-9]_[a-z,A-Z]+[0-2])'
        tempdf['collect_id'] = re.findall(pat1, path)[0]
        df = pd.concat([df, tempdf])
    df = df.sort_values(by='Time_s')
    return df.drop_duplicates()

def clean_combined_header(df, year, out_folder, dpth = 0, alt = 4, clean_lat_lon = True):
    ## Cleaning for Altitude and depth (NO LONGER USE)
    df = df[(df.Alt_m < alt) & (df.Alt_m > 0) & (df.AUV_depth_m > dpth)]
    ## Cleaning lat lon columns
    if clean_lat_lon:
        idx_bad_lat = df[(df['Lat_DD'] < 41) | (df['Lat_DD'] > 50)].index
        idx_bad_lon = df[(df['Long_DD'] < -92.5) | (df['Long_DD'] > -75.5)].index
        df.loc[idx_bad_lat, 'Lat_DD'] = np.nan
        df.loc[idx_bad_lon, 'Long_DD'] = np.nan
    df.to_pickle(os.path.join(out_folder, f"{year}_headers_combined_filtered.pickle"))
    print(year, df.shape) # (2416264, 86) (2417064, 86) (2418247, 89) (2434252, 87) (2492433, 17) (2692480, 17)
    return df

def create_unpacked_images_metatada_df_old_imn(header_df, image_df, year, ext = ".png"):
    # merging and keeping only image names that are unpacked
    image_df, header_df = image_df.copy(), header_df.copy()
    image_df.filename = image_df.filename.str.replace("CI", "PI")
    df_unp = header_df[header_df.filename.isin(image_df.filename)].copy()
    df_unp.loc[:,"year"] = df_unp.Time_s.apply(return_time().get_Y)
    df_unp.loc[:,"month"] = df_unp.Time_s.apply(return_time().get_m)
    df_unp.loc[:,"day"] = df_unp.Time_s.apply(return_time().get_d)
    df_unp.loc[:,"time"] = df_unp.Time_s.apply(return_time().get_t)
    image_df.loc[:,"old_filename"] = image_df.filename
    cam_sys = df_unp.collect_id.apply(lambda x: x.split("_")[-1])
    ims_dict_w = {"ABS1" : 4096, "ABS2": 4096, "2G": 4112}
    ims_dict_h = {"ABS1" : 2176, "ABS2": 3000, "2G": 3008}
    df_unp.loc[:, "imw"] = cam_sys.apply(lambda x: ims_dict_w.get(x, None)) 
    df_unp.loc[:, "imh"] = cam_sys.apply(lambda x: ims_dict_h.get(x, None)) 
    df_unp = pd.merge(df_unp, image_df, on=["filename", "collect_id"], how = 'inner')
    # if old_filenames:
    ofn = lambda x: "image_raw_" + x.split(".")[0].split("_")[1][-5:] + x.split(".")[0].split("_")[2] + ext
    header_df2 = header_df[header_df.Time_s <= 1609372800].copy() # before 2020
    old_filename = header_df2.filename.apply(ofn)
    header_df2["old_filename"] = old_filename.values
    df_unp2 = header_df2[header_df2.old_filename.isin(image_df.filename)]
    df_unp2 = pd.merge(df_unp2, image_df[["old_filename", "collect_id", "image_path"]], on=["old_filename", "collect_id"], how = 'inner')
    df_unp = pd.concat([df_unp, df_unp2])
    df_unp = df_unp.sort_values(by='Time_s')
    assert df_unp.shape[0] == df_unp.drop_duplicates(subset=['Time_s','filename','collect_id']).shape[0]
    df_unp.to_csv(f"all_unpacked_images_metadata_{year}.csv")
    print(year, df_unp.shape) 
    return df_unp

def create_unpacked_images_metatada_df(header_df, image_df, year, ext = ".png"):
    # merging and keeping only image names that are unpacked
    image_df, header_df = image_df.copy(), header_df.copy()
    image_df.filename = image_df.filename.str.replace("CI", "PI")
    df_unp = header_df[header_df.filename.isin(image_df.filename)].copy()
    df_unp.loc[:,"year"] = df_unp.Time_s.apply(return_time().get_Y)
    df_unp.loc[:,"month"] = df_unp.Time_s.apply(return_time().get_m)
    df_unp.loc[:,"day"] = df_unp.Time_s.apply(return_time().get_d)
    df_unp.loc[:,"time"] = df_unp.Time_s.apply(return_time().get_t)
    cam_sys = df_unp.collect_id.apply(lambda x: x.split("_")[-1])
    ims_dict_w = {"ABS1" : 4096, "ABS2": 4096, "2G": 4112}
    ims_dict_h = {"ABS1" : 2176, "ABS2": 3000, "2G": 3008}
    df_unp.loc[:, "imw"] = cam_sys.apply(lambda x: ims_dict_w.get(x, None)) 
    df_unp.loc[:, "imh"] = cam_sys.apply(lambda x: ims_dict_h.get(x, None)) 
    df_unp = pd.merge(df_unp, image_df, on=["filename", "collect_id"], how = 'inner')
    df_unp = df_unp.sort_values(by='Time_s')
    assert df_unp.shape[0] == df_unp.drop_duplicates(subset=['Time_s','filename','collect_id']).shape[0]
    df_unp.to_csv(f"all_unpacked_images_metadata_{year}.csv")
    print(year, df_unp.shape) 
    return df_unp

def plot_epoch_time(df1, df2=pd.DataFrame(),title=None, lbl1=None, lbl2=None):
    tdt = lambda x: datetime.datetime.fromtimestamp(x)
    fig, ax = plt.subplots(1, figsize=(6,3))
    et1 = df1
    c1 = np.ones(len(et1))
    dt1 = et1.apply(tdt).values
    dt1 = pd.DataFrame([dt1,c1]).T
    dt1 = dt1.set_index(0)
    counts = dt1.groupby(pd.Grouper(freq='1D')).count()
    count_values = [list for sublist in counts.values for list in sublist]
    ax.bar(counts.index, count_values, edgecolor='k', label=lbl1)
    if not df2.empty:
        et2 = df2
        c2 = np.ones(len(et2))
        dt2 = et2.apply(tdt).values
        dt2 = pd.DataFrame([dt2,c2]).T
        dt2 = dt2.set_index(0)
        counts2 = dt2.groupby(pd.Grouper(freq='1D')).count()
        count_values2 = [list for sublist in counts2.values for list in sublist]
        ax.bar(counts2.index, count_values2, edgecolor='k', alpha=0.5, label=lbl2)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

def plot_collect_distribution(df1, df2=pd.DataFrame(),title=None, lbl1=None, lbl2=None, ylabel=None):
    fig, ax = plt.subplots(1, figsize=(10,7))
    counts1 = df1.groupby(by='collect_id').count()["image_name"]
    ax.bar(counts1.index, counts1.values, edgecolor='k', label=lbl1)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    if not df2.empty:
        counts2 = df2.groupby(by='collect_id').count()["image_name"]
        ax.bar(counts2.index, counts2.values, edgecolor='k', alpha=0.5, label=lbl2)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title(title)
    ax.legend()
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return