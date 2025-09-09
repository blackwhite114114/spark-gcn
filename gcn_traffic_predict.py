from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS

# 初始化 SparkSession
spark = SparkSession.builder.appName("GCNTrafficPrediction").getOrCreate()
sc = spark.sparkContext  # 获取 SparkContext 以兼容 MLlib

# 加载图数据（CSV 格式，包含顶点和边）
vertices = spark.read.csv("/home/hadoop/projects/data/vertices.csv", header=True, inferSchema=True)
edges = spark.read.csv("/home/hadoop/projects/data/edges.csv", header=True, inferSchema=True)

# 创建图结构
graph = GraphFrame(vertices, edges)

# 模拟加载交通流数据（假设每行是 node_id, feature1, feature2, ..., label）
traffic_data = sc.textFile("/home/hadoop/projects/data/traffic_flow.txt")

# 解析为 LabeledPoint
parsed_data = traffic_data.map(lambda line: (
    int(line.split(",")[0]),  # 节点 ID
    Vectors.dense([float(x) for x in line.split(",")[1:-1]]),  # 特征向量
    float(line.split(",")[-1])  # 标签值
)).map(lambda x: LabeledPoint(x[2], x[1]))  # 转换为 LabeledPoint

# 训练逻辑回归模型模拟 GCN 行为（实际 GCN 需深度学习框架）
model = LogisticRegressionWithLBFGS.train(parsed_data, numClasses=2)

# 模型评估
labels_and_preds = parsed_data.map(lambda p: (p.label, model.predict(p.features)))
accuracy = labels_and_preds.filter(lambda v_p: v_p[0] == v_p[1]).count() / float(parsed_data.count())
print(f"Model Accuracy: {accuracy}")