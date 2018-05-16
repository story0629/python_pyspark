# -*- coding: utf-8 -*-

import os
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import *


class Utils():
    def __init__(self):
        pass

    # 敘述性統計：平均數 標準差
    def getStatValue(self, df, fieldName):
        stat = df.select(avg(fieldName), stddev(fieldName)).collect()
        return stat[0]

class loadDataCSV(Utils):
    # 繼承
    def __init__(self):
        Utils.__init__(self)

    # 載入 CSV 檔案
    def loadData(self, dataFile, schema, hasHeader=False):
        df = sqlContext.read.format('com.databricks.spark.csv').options(header=str(hasHeader)).schema(schema).load(dataFile)
        return df

    # 列印敘述性統計
    def printStats(self, df, fields=None):
        if fields is None:
            df.describe().show()
        else:
            for field in fields:
                df.describe(field).show()

# 主程式
def main(sampleDir, dataDir):
    # 資料欄位名稱
    fields = ['m1', 'm2', 'n1', 'n2', 'shape']

    # 類別初始化
    worker = loadDataCSV()

    # 指定資料綱要
    schema = StructType([
                            StructField(fields[0], DoubleType()),
                            StructField(fields[1], DoubleType()),
                            StructField(fields[2], DoubleType()),
                            StructField(fields[3], DoubleType()),
                            StructField(fields[4], StringType())
                         ])

    # 載入無欄位定義 CSV 資料，並轉換資料欄位綱要
    df = worker.loadData('%s/iris.data' % sampleDir, schema)

    # 保存資料集至指定目錄下
    df.write.mode('overwrite').save('%s/iris.parquet' % dataDir,
                                        format='parquet')

def add_mn1(m1, n1):
    return m1*n1

# 程式進入點
if __name__ == '__main__':
    global sc, sqlContext

    # 本地資源運算
    appName = 'Cup-00'
    master = 'local'

    #sc = SparkContext(conf=SparkConf().setAppName(appName).setMaster(master))

    # 取得資料庫介面
    sqlContext = SQLContext(sc)

    # 調用主程式
    homeDir = os.environ['HOME']
    dirName = 'Data'
    sampleDir = '%s/Sample' % homeDir
    dataDir = '%s/Data' % homeDir
    
    #taskControl = [True, False]
    taskControl = [False, True]

    if taskControl[0]:
        main(sampleDir, dataDir)

    if taskControl[1]:
        dataFile = "%s/iris.parquet" % dataDir
        sql = 'select m1, n1, m2,n2,shape, m1*n1 as mn1 from parquet.`%s` where m1<5 and m2>3 limit 5' % dataFile
        df = sqlContext.sql(sql).show()
