from pyspark.sql import SparkSession
from graphframes import GraphFrame
from graphframes.lib import AggregateMessages  # 用于邻居特征聚合
import numpy as np
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
from pyspark.sql.functions import col, sum as spark_sum
from datetime import datetime, timedelta
import random

# 创建Spark会话
spark = SparkSession.builder \
    .appName("SimpleGCNTrafficPrediction") \
    .getOrCreate()

# 定义4个节点的路网（环形结构）
vertices_data = [
    ("A", "Intersection A"),  # 节点A
    ("B", "Intersection B"),  # 节点B
    ("C", "Intersection C"),  # 节点C
    ("D", "Intersection D")   # 节点D
]

edges_data = [
    ("A", "B", "Road AB"),  # 边A→B
    ("B", "C", "Road BC"),  # 边B→C
    ("C", "D", "Road CD"),  # 边C→D
    ("D", "A", "Road DA")   # 边D→A（形成环形）
]

# 创建顶点DataFrame
vertices_schema = StructType([
    StructField("id", StringType(), False),
    StructField("description", StringType(), True)
])

vertices_df = spark.createDataFrame(vertices_data, schema=vertices_schema)

# 创建边DataFrame并添加交通流量数据
edges_schema = StructType([
    StructField("src", StringType(), False),
    StructField("dst", StringType(), False),
    StructField("road_name", StringType(), True)
])

edges_df = spark.createDataFrame(edges_data, schema=edges_schema)

# 生成48小时（2天）的时间序列交通流量数据
base_flows = {"A": 100, "B": 150, "C": 200, "D": 120}  # 不同边的基础流量
time_based_flows = []
start_time = datetime(2023, 1, 1, 0, 0, 0)

for hour in range(48):  # 48小时
    current_time = start_time + timedelta(hours=hour)
    
    # 创建具有明显时间模式的流量（早晚高峰、深夜低谷）
    if 7 <= current_time.hour <= 9 or 17 <= current_time.hour <= 19:  # 高峰时段
        time_factor = 2.0
    elif 22 <= current_time.hour <= 23 or 0 <= current_time.hour <= 5:  # 深夜低谷
        time_factor = 0.5
    else:
        time_factor = 1.0  # 正常时段
    
    for src in ["A", "B", "C", "D"]:
        # 计算随机波动（±20%）
        random_factor = 1.0 + (random.random() - 0.5) * 0.4
        
        # 计算当前小时的流量
        flow = base_flows[src] * time_factor * random_factor
        
        time_based_flows.append([
            src,
            hour,
            current_time.strftime("%Y-%m-%d %H:%M:%S"),
            flow
        ])

# 创建时间特征DataFrame
time_features_schema = StructType([
    StructField("src", StringType(), False),
    StructField("hour", IntegerType(), False),
    StructField("timestamp", StringType(), True),
    StructField("flow", DoubleType(), False)
])

time_features_df = spark.createDataFrame(time_based_flows, schema=time_features_schema)

# 创建基础图结构
graph = GraphFrame(vertices_df, edges_df)

# 添加交通流量特征到顶点（取平均值作为初始特征）
features_df = time_features_df.groupBy("src").avg("flow").withColumnRenamed("avg(flow)", "features")
vertices_with_features_df = vertices_df.join(features_df, vertices_df.id == features_df.src, "inner") \
    .select(vertices_df["id"], "description", "features")

# 重新创建带特征的图
graph_with_features = GraphFrame(vertices_with_features_df, edges_df)

# 显示图的初始状态
print("🌐 初始图结构:")
graph_with_features.vertices.show()
graph_with_features.edges.show()

# 显示时间序列流量数据
print("📈 时间序列流量数据:")
time_features_df.show()

# 定义邻居聚合函数
def aggregate_neighbors(g):    
    # 使用AggregateMessages进行邻居特征聚合
    agg = g.aggregateMessages(
        spark_sum(AggregateMessages.msg).alias("aggregatedMsg"),
        sendToDst=col("src.features")
    )
    
    # 更新节点特征（当前特征 + 聚合特征）
    updated_vertices = g.vertices.join(agg, g.vertices.id == agg.id, "left_outer") \
        .select(
            g.vertices["id"],
            "description",
            (g.vertices["features"] + agg["aggregatedMsg"]).alias("features")
        )
    
    # 创建新的图结构
    return GraphFrame(updated_vertices, g.edges)

# 模拟多层GCN的信息传播
num_iterations = 3  # 模拟3层GCN
current_graph = graph_with_features

for i in range(num_iterations):
    print(f"🔄 第 {i+1} 层GCN信息传播...")
    current_graph = aggregate_neighbors(current_graph)
    print(f"📊 当前节点特征:")
    current_graph.vertices.show()

# 停止Spark会话
spark.stop()
