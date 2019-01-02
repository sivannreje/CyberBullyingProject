#!/usr/local/bin/python3
# The main code cell
# this cell should be hide
import pandas as pd
import numpy as np
import csv
import glob
import pickle
import gc
import configFile

hebABC = 'אבגדהוזחטיכלמנסעפצקרשת'
hebFinal = 'ןםףךץ'
hebAll = hebABC + hebFinal
stop_words = None

def saveObj(path,obj):
    with open(path,'wb') as pickle_out:
        pickle.dump(obj,pickle_out)

def loadObj(path):
    with open(path,'rb') as pickle_in:
        obj = pickle.load(pickle_in)
    return obj

def csvToDic(path,encoding,header=True):
    try:
        with open(path, mode='r',encoding=encoding) as infile:
            reader = csv.reader(infile)
            if header:
                reader.__next__()
            mydict = {rows[0]: rows[1] for rows in reader}
    except UnicodeDecodeError as e:
        with open(path, mode='r',encoding='utf-8') as infile:
            reader = csv.reader(infile)
            if header:
                reader.__next__()
            mydict = {rows[0]: rows[1] for rows in reader}
    return mydict

def traverse(a):
    if type(a) is list:
        return [''.join(wrd[-1:-(len(wrd)+1):-1]) if type(wrd) is str and len(wrd)>0 and wrd[0] in 'אבגדהוזחטיכלמנסעפצקרשת' else wrd for wrd in a ]
    elif type(a) is str: return traverse([a])[0]
    elif type(a) is set: return set(traverse(list(a)))
    elif type(a) is dict: dict(zip(traverse(a.keys()),traverse(a.values())))
    elif type(a) == type(pd.Series()): return pd.Series(data=traverse(list(a)),index=a.index,name=a.name)
    elif type(a) == type(type(pd.DataFrame())): return a.applymap(lambda x: traverse(x))
    return a

def fileToList(path, encoding='cp1255',header=True):
    try:
        with open(path, mode='r',encoding=encoding) as infile:
            myList = [line.strip('\n') for line in infile]
    except UnicodeDecodeError as e:
        with open(path,mode='r', encoding='utf-8') as infile:
            myList = [line.strip('\n') for line in infile]
    return myList

alpha_numeric_dic = {}

def init_alpha_numeric_dic():
    global alpha_numeric_dic
    for i in range(ord('a'), ord('z') + 1):
        alpha_numeric_dic[chr(i)] = True
    for i in range(ord('A'), ord('Z') + 1):
        alpha_numeric_dic[chr(i)] = True
    for i in range(ord('0'), ord('9') + 1):
        alpha_numeric_dic[chr(i)] = True

def folderToDF(dir_path):
    allFiles = glob.glob(dir_path+"/*.csv")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, index_col=None, header=0,encoding='cp1255')
        list_.append(df)
    return pd.concat(list_)

def discretize(data, bins):
    split = np.array_split(np.sort(data), bins)
    cutoffs = [x[-1] for x in split]
    cutoffs = cutoffs[:-1]
    discrete = np.digitize(data, cutoffs, right=True)
    return discrete  #, cutoffs

def reduce(f,iterable,start=0):
    result = start
    for x in iterable:
        result = f(result,x)
    return result

init_alpha_numeric_dic()

def genertate_small_df_from_big_df(path, df_size, cols=None):
    splits_size = configFile.SPLITS_SIZE
    for i in range(splits_size):
        start = int((i/splits_size)*df_size)
        end = int(((i+1)/splits_size)*df_size)
        df = pd.read_csv(path,encoding='cp1255',skipinitialspace=True, usecols=cols,skiprows=range(1,start),nrows=end-start)
        yield df
        gc.collect()

def write_to_log(msg):
    pass
