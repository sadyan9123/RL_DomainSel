import pandas as pd
import numpy as np
import pickle
import time
from sklearn import preprocessing
from sklearn.externals import joblib
import os
from enum import Enum
import  collections
import time

from .PandasHelper import *


dir_path = os.path.dirname(os.path.realpath(__file__))
##print(dir_path)
TRAIN_PORCESSING_MODEL_PATH = dir_path + '/../output/pro_models/'
##print(TRAIN_PORCESSING_MODEL_PATH)
ZSCORE_FILE = "zscore.pkl"
MINMAX_FILE = "minmax.pkl"
CUTV_FILE = "cutvalue.pkl"
CUTD_FILE = "cutdensity.pkl"
BUCKET_FILE = "BucketCutModel.pkl"
BUCKETP_FILE = "BucketPCutModel.pkl"
PROCESSING_FILE = "ProcessingMethods.pkl"
DELETEDCOLS_FILE = "DeletedCols.pkl"
FULL_FEATURE_PREFIX = "full"

IS_FULL_FEATURE = False

def getFeatureNameList():
    FEATURE_FILE_NAME = open('FEATURE_STATIC_25.txt').readlines()
    FEATURE_DOMAINSEL_NAME = open('FEATURE_VA-PA_72.txt').readlines()
    FEATURE_FILE_NAME = [s.strip() for s in FEATURE_FILE_NAME]
    FEATURE_DOMAINSEL_NAME = [s.strip() for s in FEATURE_DOMAINSEL_NAME]
    FEATURE_FILE_NAME.extend(FEATURE_DOMAINSEL_NAME)
    FEATURE_FILE_NAME.append("Min type score extend")
    return FEATURE_FILE_NAME


def getFeatureFileNames():
    lines = open('FEATURE_STATIC_25.txt').readlines()
    names = [s.strip() for s in lines]
    return names

def generateProgramData(trans_seq):
    total = []
    cur = []
    for i, row in enumerate(trans_seq):
        cur.append(row[0])
        if row[-1] == 1: #terminal is true
            total.append(cur.copy())
            cur.clear()
    return total

def convertToTotalTraces(data):
    res = []
    for program_trace in data:
        res.extend(program_trace)

    return res

def saveTempPandasFile(data, timestamp):
    tdata = convertToTotalTraces(data)
    names = getFeatureNameList()
    df = pd.DataFrame(tdata, columns=names)
    df.to_csv("data" + timestamp + ".csv", index=False)


def dropColsWithOnlyOneValue(df):
    cols = []
    for col in df.columns:
        if len(df[col].unique()) == 1:
            cols.append(col)

    df.drop(cols, inplace=True, axis=1)
    ##print("drop %d columns with only unique value" % (len(cols)))
    return cols



def processDataSetsByRules(df, min_max_cols):
    no_processing_cols = list()
    #min_max_cols = list()
    zscore_cols = list()
    log_cols = list()
    bucket_cols = list()
    concrete_algorithms = ProcessingAlgorithm(True)
    add_to = {
        ProcessingMethod.NO_PM : (no_processing_cols, concrete_algorithms.noProcessing),
        ProcessingMethod.LOG_PM : (log_cols, concrete_algorithms.transformByLog),
        ProcessingMethod.BUCKET_PM : (bucket_cols, concrete_algorithms.doBucketing),
        ProcessingMethod.ZSCORE_PM : (zscore_cols, concrete_algorithms.standariztionByZ_score),
        ProcessingMethod.MIN_MAX_PM : (min_max_cols, concrete_algorithms.minMaxScale),
        ProcessingMethod.PBUCKET_PM: ([], concrete_algorithms.doPBucketing)
    }

    choose = AlgorithmChoose()
    for col_name in df.columns:
        if col_name in min_max_cols:
            continue
        global IS_FULL_FEATURE
        if not IS_FULL_FEATURE:
            method = choose.getPreferredMethod(df[col_name], col_name)
            add_to[method][0].append(col_name)
        else:
            if choose.getPreferredMethod(df[col_name], col_name) is not ProcessingMethod.NO_PM:
                for m in (ProcessingMethod.LOG_PM, ProcessingMethod.BUCKET_PM,
                          ProcessingMethod.ZSCORE_PM, ProcessingMethod.PBUCKET_PM):
                    add_to[m][0].append(col_name)
            else:
                add_to[ProcessingMethod.NO_PM][0].append(col_name)

    return add_to, concrete_algorithms


class ProcessingMethod(Enum):
    NO_PM = 0
    LOG_PM = 1
    BUCKET_PM = 2
    ZSCORE_PM = 3
    MIN_MAX_PM = 4
    PBUCKET_PM = 5

class AlgorithmChoose():

    def __init__(self):
        pass

    def getPreferredMethod(self, col, name):
        if self.shoudNotProcess(col):
            return ProcessingMethod.NO_PM
        elif self.shouldUseLog(col, name):
            return ProcessingMethod.LOG_PM
        elif self.shouldUseBuecktTransform(col):
            return ProcessingMethod.BUCKET_PM
        else:
            return ProcessingMethod.ZSCORE_PM


    def shouldUseBuecktTransform(self, col):
        _max = col.max()
        _min = col.min()
        if _min < 0 or _max < 90:
            return False
        return True

    def shouldUseLog(self, col, name):
        if 'Time' in name and 'Times' not in name:
            return True
        else:
            return False

    def shoudNotProcess(self, col):
        _max = col.max()
        _min = col.min()
        if _min == 0 and _max == 1:
            return True
        else:
            return False


def saveModelToFile(model, file_name):
    joblib.dump(model, os.path.join(TRAIN_PORCESSING_MODEL_PATH, file_name))

def loadModelFromFile(file_name):
    return joblib.load(os.path.join(TRAIN_PORCESSING_MODEL_PATH, file_name))


class ProcessingAlgorithm():

    def __init__(self, is_train):
        self.is_train = is_train
        self.zscore_params = {}
        self.bucket_params = {}
        self.minmax_params = {}
        self.pbucket_params = {}
        if not is_train:
            self.zscore_params = readParamsFromPKL(ZSCORE_FILE)
            self.bucket_params = readParamsFromPKL(BUCKET_FILE)
            self.minmax_params = readParamsFromPKL(MINMAX_FILE)
            self.pbucket_params = readParamsFromPKL(BUCKETP_FILE)


    def saveParams(self):
        saveParamsToPKL(self.zscore_params, ZSCORE_FILE)
        saveParamsToPKL(self.bucket_params, BUCKET_FILE)
        saveParamsToPKL(self.minmax_params, MINMAX_FILE)
        saveParamsToPKL(self.pbucket_params,BUCKETP_FILE)

    def noProcessing(self, X, name):
        return X


    #they might behave badly if the individual features do not more or less look like standard
    # normally distributed data: Gaussian with zero mean and unit variance
    def standariztionByZ_score(self, X, name):
        if self.is_train:
            scalar = preprocessing.StandardScaler().fit(X)
            self.zscore_params[name] = scalar
        else:
            scalar = self.zscore_params[name]
        return scalar.transform(X)


    def transformByLog(self, X, name):
        return preprocessing.FunctionTransformer(np.log1p,validate=False).transform(X)

    def minMaxScale(self, X, name):
        if self.is_train:
            scalar = preprocessing.MinMaxScaler().fit(X)
            self.minmax_params[name] = scalar
        else:
            scalar = self.minmax_params[name]
        return scalar.transform(X)

    def doBucketing(self, X, name):

        if self.is_train:
            scalar =  preprocessing.KBinsDiscretizer(n_bins=10, encode='ordinal').fit(X)
            self.bucket_params[name] = scalar
        else:
            scalar = self.bucket_params[name]
        return scalar.transform(X)

    def doPBucketing(self, X, name):
        if self.is_train:
            scalar = preprocessing.KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile').fit(X)
            self.pbucket_params[name] = scalar
        else:
            scalar = self.pbucket_params[name]
        return scalar.transform(X)


def getBinsIndexMap(bins):
    bins_maps = {}
    for pair in adjacent_pairs(bins):
        bins_maps[pair] = len(bins_maps)
    return bins_maps

def adjacent_pairs(seq):
    it = iter(seq)
    prev = next(it)
    for item in it:
        yield (prev, item)
        prev = item

def saveParamsToPKL(data, file_name):
    global  IS_FULL_FEATURE
    if IS_FULL_FEATURE:
        file_name = FULL_FEATURE_PREFIX + file_name
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

def readParamsFromPKL(file_name):
    global IS_FULL_FEATURE
    if IS_FULL_FEATURE:
        file_name = FULL_FEATURE_PREFIX + file_name
    exists = os.path.isfile(file_name)
    if exists:
        return pickle.load(open(file_name, 'rb'))
    else:
        return {}


def startProcessingInTrain(df, processing_info):
    results = []
    saved_cols_method = {}
    for method_name, info in processing_info.items():
        col_names, fun = info
        if not col_names:
            continue
        ##print(col_names)
        saved_cols_method[method_name] = col_names
        data = fun(df[col_names].values, ','.join(col_names))
        data = pd.DataFrame(data, columns=col_names)
        results.append(data)
        ##print("method %s has applied to %d features" % (method_name, len(col_names)))

    saveParamsToPKL(saved_cols_method, PROCESSING_FILE)
    return pd.concat(results, axis=1).values
    #return np.asarray(results).transpose()



def saveFromNpToPandas(data, timestamp):
    #name may have duplicate
    names = getFeatureNameList()
    df = pd.DataFrame(data, columns=names)
    df.to_csv("data" + str(timestamp) + ".csv", index=False)

def getFeaturedTrainData(data, is_full_feature):
    global IS_FULL_FEATURE
    IS_FULL_FEATURE = is_full_feature
    #print(len(data[0]))
    timestamp = time.time()
    saveFromNpToPandas(data, timestamp)
    df = pd.read_csv("data" + str(timestamp) + ".csv")
    names = getFeatureNameList()
    unique_names = set()
    droped_index = []
    for idx, name in enumerate(names):
        if name in unique_names:
            droped_index.append(idx)
        else:
            unique_names.add(name)
    df.drop(df.columns[droped_index], inplace=True, axis=1)
    #print("droped {} columns according to names in train".format(len(droped_index)))
    deleted_cols = []
    deleted_cols.extend(dropColsWithOnlyOneValue(df))
    duplicate_cols = getDuplicateColumns(df)
    deleted_cols.extend(duplicate_cols)
    saveParamsToPKL(deleted_cols, DELETEDCOLS_FILE)
    df.drop(columns=duplicate_cols, inplace=True, axis=1)
    #print("drop %d duplicate colmns" % (len(duplicate_cols)))
    df.to_csv("DataAnalaysis.csv", index=False)
    min_max_cols = []

    process_info, algo = processDataSetsByRules(df, min_max_cols)
    res = startProcessingInTrain(df, process_info)
    algo.saveParams()


    keep = 0
    ff_names = getFeatureFileNames()
    for f_name in ff_names:
        if f_name in df.columns:
            keep += 1
    #return res.values.tolist()

    return res.tolist(),keep

def getFeaturedSingleTestData(test_data, is_full_feature):
    #start_time = time.time()
    global IS_FULL_FEATURE
    IS_FULL_FEATURE = is_full_feature
    names = getFeatureNameList()
    unique_names = set()
    droped_index = []
    for idx, name in enumerate(names):
        if name in unique_names:
            droped_index.append(idx)
            del test_data[idx]
            del names[idx]
        else:
            unique_names.add(name)
    #print("droped {} columns according to names in test".format(len(droped_index)))
    deleted_cols = readParamsFromPKL(DELETEDCOLS_FILE)
    data = []
    data_names = []
    name_to_v = {}
    for v, name in zip(test_data, names):
        if name not in deleted_cols:
            data.append(v)
            data_names.append(name)
            name_to_v[name] = v

    processing_info = readParamsFromPKL(PROCESSING_FILE)
    #ans = [0] * len(data)

    names_to_v = {}
    ans = []
    algorithms = ProcessingAlgorithm(False)
    for method, names in processing_info.items():
        if not names:
            continue
        lnames = names.copy()
        names = ','.join(names)
        if names not in names_to_v:
            value = [[]]
            for name in lnames:
                value[0].append(name_to_v[name])
            names_to_v[names] = value
        value = names_to_v[names]
        if method == ProcessingMethod.NO_PM:
            res = value
        elif method == ProcessingMethod.MIN_MAX_PM:
            res = algorithms.minMaxScale(value, names)
        elif method == ProcessingMethod.ZSCORE_PM:
            res = algorithms.standariztionByZ_score(value, names)
        elif method == ProcessingMethod.BUCKET_PM:
            res = algorithms.doBucketing(value, names)
        elif method == ProcessingMethod.PBUCKET_PM:
            res = algorithms.doPBucketing(value, names)
        else:
            res = algorithms.transformByLog(value, names)
        ans.extend(res[0])
    ##print("--- %s seconds ---" % (time.time() - start_time))
    return ans



#testTrain("1554466495.5501351")


