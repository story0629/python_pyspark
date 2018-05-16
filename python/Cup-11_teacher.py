# -*- coding: utf-8 -*-

import os
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


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

    # 決策樹
    def DT(self, trainingData, testData, labelIndexer, features):
        # 組合自變數欄位群，並指明衍生欄位名稱
        features = (VectorAssembler()
                        .setInputCols(features)
                        .setOutputCol('features'))

        # 取得決策樹介面
        dt = DecisionTreeClassifier(labelCol='indexedLabel', featuresCol='features')

        # 進行決策樹分析
        pipeline = Pipeline(stages=[labelIndexer, features, dt])

        # 產生決策樹分析模型
        model = pipeline.fit(trainingData)

        # 推測值
        predictions = model.transform(testData)

        return predictions

    # 列印決策樹分析結果
    def printStatsDT(self, predictions):
        # 篩選分析結果欄位群
        result = predictions.select('indexedLabel', 'prediction', 'features', 'probability')

        # 篩選預測錯誤資料
        resultError = result.where(result.indexedLabel != result.prediction)
        resultError.show()

        print(u'準確率=%.3f (%d\t%d)' % (1.000 - resultError.count() / result.count(),
                resultError.count(),
                result.count()))


# 主程式
def main(dataDir):
    # 類別初始化
    worker = LoadSavedData()

    # 載入資料集
    df = worker.loadData(dataFile='%s/iris.parquet' % dataDir)

    # 資料隨機抽樣成二群
    (trainingData, testData) = df.randomSplit([0.7, 0.3])

    # 為類別值建立數值對照表
    labelIndexer = StringIndexer(inputCol='shape', outputCol='indexedLabel').fit(df)

    # 決策樹：指定自變數欄位群
    result = worker.DT(trainingData, testData, labelIndexer, df.columns[0:4])
    result.printSchema()

    # 列印決策樹分析結果
    worker.printStatsDT(result)

# 程式進入點
if __name__ == '__main__':
    global sc, sqlContext

    # 本地資源運算
    appName = 'Cup-11'
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
