# -*- coding: UTF-8 -*-
from pyspark.mllib.recommendation import ALS
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

def SetLogger( sc ):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)    

def SetPath(sc):
    global Path
    Path = ""

def PrepareData(sc): 
    #----------------------1.建立使用者評價資料-------------
    print("開始讀取使用者評價資料中...")
    rawUserData = sc.textFile(Path+"data/u.data")
    rawRatings = rawUserData.map(lambda line: line.split("\t")[:3] )
    ratingsRDD = rawRatings.map(lambda x: (x[0],x[1],x[2]))
    #----------------------2.顯示資料筆數-------------
    numRatings = ratingsRDD.count()
    numUsers = ratingsRDD.map(lambda x: x[0] ).distinct().count()
    numMovies = ratingsRDD.map(lambda x: x[1]).distinct().count() 
    print("共計：ratings: " + str(numRatings) +    
             " User:" + str(numUsers) +  
             " Movie:" +    str(numMovies))
    return(ratingsRDD)

def SaveModel(sc): 
    try:        
        model.save(sc,Path+"model")
        print("已儲存Model 在model")
    except Exception as e:
        print("Model已經存在,請先刪除再儲存.")
        print(e)
    
if __name__ == "__main__":
    #sparkConf = SparkConf().setAppName("RecommendTrain").set("spark.ui.showConsoleProgress", "false")
    #sc = SparkContext(conf = sparkConf)
    spark = SparkSession.builder.getOrCreate()
    print("master="+sc.master)    
    SetLogger(sc)
    SetPath(sc)
    print("==========資料準備階段===========")
    ratingsRDD = PrepareData(sc)
    print("==========訓練階段===============")
    print("開始ALS訓練,參數rank=5,iterations=20, lambda=0.1");
    model = ALS.train(ratingsRDD, 5, 20, 0.1)
    print("========== 儲存Model========== ==")
    SaveModel(sc)

