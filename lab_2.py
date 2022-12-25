from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.mllib.linalg import Vectors
import numpy as np

conf = SparkConf().setMaster("local[*]").setAppName("SparkTFIDF")
sc = SparkContext(conf=conf)

spark = SparkSession.builder.getOrCreate()

df = sc.textFile('ml-20mx16x32.csv').map(lambda x: map(int, x.split())).groupByKey().map(lambda x : (x[0], list(x[1]))).toDF(schema=["userId", "movieId"])

hashingTF = HashingTF(inputCol="movieId", outputCol="rawFeatures", numFeatures=10000)
tf = hashingTF.transform(df)
tf.cache()
idf = IDF(inputCol="rawFeatures", outputCol="features").fit(tf)
tfidf = idf.transform(tf)

uniq_users = np.unique(df.select('userId').distinct().collect())
user_id = np.random.choice(uniq_users, 1)[0]

matrix = IndexedRowMatrix(tfidf.select("userId", "features").rdd.map(lambda row: IndexedRow(row.userId, row.features.toArray()))).toBlockMatrix()
cos_matrix = matrix.transpose().toIndexedRowMatrix().columnSimilarities()
filter_cos = cos_matrix.entries.filter(lambda x: x.i == user_id or x.j == user_id)

sort_sim = filter_cos.sortBy(lambda x: x.value, ascending=False).map(lambda x: IndexedRow(x.j if x.i == user_id else x.i,  Vectors.dense(x.value)))

users = []
for row in sort_sim.collect():
    users.append(row.index)

count = np.random.randint(10, 100)

test_user = df.filter(df.userId == users[0]).select("movieId").rdd
test_user = set(test_user.collect()[0].movieId)
test_id = df.filter(df.userId == user_id).select("movieId").rdd
test_id = set(test_id.collect()[0].movieId)
result = list(test_user - set.intersection(test_user, test_id))
if len(result) >= count:
    print(result[:count])
else:
    print(result)

spark.stop()