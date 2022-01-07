
import petl as etl
import glob
import pandas as pd



# get data file names
path =r'F:/PROJECTS/SSIS/Data/'
filenames = glob.glob(path + "/*.xlsx")

dfs = []
for filename in filenames:
    dfs.append(pd.read_excel(filename))

# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)
big_frame.drop_duplicates(subset=['id'])