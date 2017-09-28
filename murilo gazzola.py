##################################################
#Librarys
##################################################
#arrays
import numpy as np

#data extraction
import pandas as pd

#xgboost
import xgboost as xgb

#Sklearn
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

local_train="../../puzzle/puzzle_train_dataset.csv"
local_test="../../puzzle/puzzle_test_dataset.csv"    
    
def get_extract(): #features and label
    # load dataset in folder
    dataframe_train = pd.read_csv(local_train, header=None)


    ##conversion required for integer, remove infinites and solve float32 problem
    dataframe_train = dataframe_train[:].convert_objects(convert_numeric=True).fillna(0)
    
    #Values out Dataframe.Panda
    dataset_train = dataframe_train.values
    
    dataset_train[:,15]=(dataset_train[:,15]=="f").astype(int) #Boolean columns converted to integer - genre
    dataset_train[:,16]=(dataset_train[:,16]).astype(int) #value boolean is facebook

    X=dataset_train[1:,[4,5,6,7,8,9,10,11,13,15,24,25,26,27]] # Removed columns that are not converted to learning
    y=(dataset_train[1:,1]).astype(int) #bug fix
    
    return X,y

def get_extract_test():
    dataframe_test = pd.read_csv(local_test, header=None)
    dataframe_test = dataframe_test[:].convert_objects(convert_numeric=True).fillna(0)
    dataset_test = dataframe_test.values

    dataset_test[:,14]=(dataset_test[:,14]=="f").astype(int) #Test column -1 (default)
    
    X_validation_db=dataset_test[1:,[3,4,5,6,7,8,9,10,12,14,23,24,25,26]] # Least one because of the default removed on the test portion
    

    
    return X_validation_db


def train_model_xgboost():
    model = xgb.XGBRegressor(max_depth=9,
                             n_estimators=1000,
                             min_child_weight=10,
                             gamma=0,
                             objective= 'binary:logistic',
                             scale_pos_weight=1,
                             learning_rate=0.1,
                             nthread=8,
                             subsample=0.80,
                             colsample_bytree=0.80,
                             seed=2017)
    X,y=get_extract()

    
    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(X, y, random_state=42, stratify=y,test_size=0.10)
    model.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=15)
    return model
    
def make_submit():    
    
    
    #XGboost
    model_xgboost=train_model_xgboost()
    pred_xgboost = model_xgboost.predict(get_extract_test())
    
    #Save
    df = pd.read_csv(local_test) #use ids

    new_df_xgb = df[['ids']] #new colums - remove others
    new_df_xgb['prediction']=pred_xgboost #predictions
    new_df_xgb.to_csv('model_xgboost.csv', index=False)
    

if __name__=='__main__':
    make_submit()