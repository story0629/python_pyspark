# -*- coding: utf-8 -*-

import os
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import *
import numpy

class Utils():
    def __init__(self):
        pass

    # 敘述性統計：平均數 標準差
    def getStatValue(self, df, fieldName):
        stat = df.select(avg(fieldName), stddev(fieldName)).collect()
        return stat[0]

class LoadSavedData(Utils):
    # 繼承
    def __init__(self):
        Utils.__init__(self)

    # 載入資料集檔案
    def loadData(self, dataFile):
        sql = 'SELECT * FROM parquet.`%s`' % dataFile
        df = sqlContext.sql(sql)
        return df

    # 列印敘述性統計
    def printStats(self, df, fields=None):
        if fields is None:
            df.describe().show()
        else:
            for field in fields:
                df.describe(field).show()

# 對照轉換欄位相應值函數
def changeValue(shape):
    idShape = 0
    if shape == 'Iris-versicolor':
        idShape = 1
    elif shape == 'Iris-virginica':
        idShape = 2

    return idShape

# 主程式
def main(dataDir):
    # 資料欄位名稱
    fields = ['m1', 'm2', 'n1', 'n2', 'shape']

    # 類別初始化
    worker = LoadSavedData()

    # 載入資料集
    df = worker.loadData(dataFile='%s/iris.parquet' % dataDir)

    # 萃取欄位不重複值
    shapes = df.select(fields[4]).distinct().collect()
    for shape in shapes:
        print(shape[0])

    # 自訂函數：對照轉換欄位相應值
    myUdf = udf(changeValue, IntegerType())

    # 對照轉換欄位相應值，衍生新欄位，取代原資料集
    df = df.withColumn('label', myUdf('shape'))

    df.show()

    # 萃取衍生欄位不重複值
    idShapes = df.select('label').distinct().collect()
    for idShape in idShapes:
        print(idShape[0])

    # 列印敘述性統計：平均數 標準差
    for field in fields[1:4]:
        stat = worker.getStatValue(df, field)
        print('%8s\t%.3f\t%.3f' % (field, stat[0], stat[1]))

    # 列印敘述性統計
    worker.printStats(df, fields[1:4])

    # 保存資料集至指定目錄下
    df.write.mode('overwrite').save('%s/iris2.parquet' % dataDir,
                                        format='parquet')


# 程式進入點
if __name__ == '__main__':
    global sc, sqlContext

    # 本地資源運算
    appName = 'Cup-01'
    master = 'local'

    #sc = SparkContext(conf=SparkConf().setAppName(appName).setMaster(master))

    # 取得資料庫介面
    sqlContext = SQLContext(sc)

    # 調用主程式
    homeDir = os.environ['HOME']
    dirName = 'Data'
    sampleDir = '%s/Sample' % homeDir
    dataDir = '%s/Data' % homeDir

    main(dataDir)
