import full_tool

######## apply to full dataset in sequence #######

## set up time recording for training
import time
start_time = time.time()


Path = "C:/Users/MuellerJona/Desktop/data_cleaner/missing_value_data_cleaner_categorical/data/EXPORT_M0P_060_BI.XLSX"

# read data   
d = full_tool.dataReader(True, False, path = Path)


# get columns with missing values
cols = d.columns
colsMissings = []

# get all columns with more than one missing value
for i in range(len(cols)):
    if d[cols[i]].isnull().sum() > 2:
        colsMissings.append(cols[i])

print("Columns with missing values: ", colsMissings)

# train models for these columns, save models, save train and test data and model predictions for missing values

for i in range(len(colsMissings)):
    helper = full_tool.dataCleaner(colsMissings[i], Path, True, False, True, True, False, cols[i] + ".sav", True, hC = True, correlation = True)

minutes = (time.time() - start_time)/60
print("--- %s minutes ---" % minutes)
   
# pip install scikit-learn==0.22.2
# 
