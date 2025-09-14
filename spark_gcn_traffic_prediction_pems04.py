# -*- coding
# @Time    : 2024/6/20 16:00
# @Author  : xiaolu
# @github  :            
# @File    : spark_gcn_traffic_prediction_pems04.py
# @Description: 使用Spark和GraphFrames实现GCN进行交通流预测，基于PeMS04数据集
# @Reference: 参考了多个资料和示例代码，结合实际需求进行实现
# 注意：需要在Spark环境中运行，并确保GraphFrames库可用


import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
try:
    from graphframes import GraphFrame
    HAS_GRAPHFRAMES = True
except ImportError:
    print("GraphFrame not available, will use basic Spark operations")
    HAS_GRAPHFRAMES = False
import h5py

def setup_spark():
    """
    设置Spark会话
    """
    # 根据经验教训，使用最简单的初始化方式
    try:
        spark = SparkSession.builder \
            .appName("GCN Traffic Prediction PeMS04") \
            .master("local[*]") \
            .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.1-s_2.12") \
            .getOrCreate()
    except:
        # 如果上面的方式失败，则使用基础配置
        spark = SparkSession.builder \
            .appName("GCN Traffic Prediction PeMS04") \
            .master("local[2]") \
            .config("spark.sql.warehouse.dir", "/tmp") \
            .config("spark.driver.host", "localhost") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.1-s_2.12") \
            .getOrCreate()
    
    # 设置检查点目录
    spark.sparkContext.setCheckpointDir("/tmp/spark-checkpoint")
    
    return spark

def load_pems04_data(spark):
    """
    加载PeMS04数据集：路网结构和交通流数据
    :param spark: SparkSession
    :return: graph, flow_data
    """
    # 加载路网结构数据
    edges_df = spark.read.option("header", "true").csv("/home/hadoop/GCN_predict-Pytorch/PeMS_04/PeMS04.csv")
    edges_df = edges_df.withColumn("src", col("from").cast("int")) \
                       .withColumn("dst", col("to").cast("int")) \
                       .withColumn("weight", col("cost").cast("float")) \
                       .select("src", "dst", "weight")
    
    # 从边数据中提取节点
    nodes_from_src = edges_df.select(col("src").alias("id"))
    nodes_from_dst = edges_df.select(col("dst").alias("id"))
    nodes_df = nodes_from_src.union(nodes_from_dst).distinct()
    
    # 创建GraphFrame
    graph = None
    global HAS_GRAPHFRAMES
    if HAS_GRAPHFRAMES:
        try:
            graph = GraphFrame(nodes_df, edges_df)
        except Exception as e:
            print(f"Could not create GraphFrame: {e}")
            HAS_GRAPHFRAMES = False
    
    # 加载交通流数据
    flow_data_np = np.load("/home/hadoop/GCN_predict-Pytorch/PeMS_04/PeMS04.npz")
    flow_data = flow_data_np['data']  # 形状为 (16992, 307, 3)
    # 只使用第一个特征（交通流量），忽略其他特征
    flow_data = flow_data[:, :, 0:1].astype(np.float32)  # 形状为 (16992, 307, 1)
    # 调整维度顺序以匹配代码期望的格式 [N, T, D]
    flow_data = np.transpose(flow_data, (1, 0, 2))  # 形状为 (307, 16992, 1)
    
    return graph, flow_data

def preprocess_flow_data(flow_data):
    """
    预处理流量数据，进行归一化
    :param flow_data: 流量数据 numpy array [N, T, D]
    :return: norm_base, norm_data
    """
    norm_dim = 1  # 在时间维度上归一化
    max_data = np.max(flow_data, axis=norm_dim, keepdims=True)
    min_data = np.min(flow_data, axis=norm_dim, keepdims=True)
    norm_base = [max_data, min_data]
    
    mid = min_data
    base = max_data - min_data
    normalized_data = (flow_data - mid) / base
    
    return norm_base, normalized_data

def slice_data(data, history_length, index, train_mode):
    """
    根据历史长度和索引划分数据样本
    :param data: np.array, normalized traffic data
    :param history_length: int, length of history data to be used
    :param index: int, index on temporal axis
    :param train_mode: str, ["train", "test"]
    :return:
        data_x: np.array, [N, H, D]
        data_y: np.array [N, D]
    """
    if train_mode == "train":
        start_index = index
        end_index = index + history_length
    elif train_mode == "test":
        start_index = index - history_length
        end_index = index
    else:
        raise ValueError("train model {} is not defined".format(train_mode))

    # 检查索引是否超出范围
    if start_index < 0 or end_index >= data.shape[1]:
        raise IndexError("Index out of bounds: start_index={}, end_index={}, data shape={}".format(
            start_index, end_index, data.shape))

    data_x = data[:, start_index: end_index]
    data_y = data[:, end_index]

    return data_x, data_y

def create_dataset(flow_data, divide_days, time_interval, history_length, train_mode):
    """
    创建训练或测试数据集
    :param flow_data: np.array, 流量数据
    :param divide_days: list, [训练天数, 测试天数]
    :param time_interval: int, 时间间隔(分钟)
    :param history_length: int, 历史数据长度
    :param train_mode: str, ["train", "test"]
    :return: 数据集
    """
    # PeMS04数据每5分钟一个数据点
    train_days = divide_days[0]
    test_days = divide_days[1]
    one_day_length = int(24 * 60 / time_interval)
    
    if train_mode == "train":
        dataset_length = train_days * one_day_length - history_length
    elif train_mode == "test":
        # 确保测试数据集不会超出数据范围
        max_test_index = flow_data.shape[1] - history_length - 1
        max_possible_tests = max_test_index - train_days * one_day_length
        # 使用条件判断代替min函数避免与PySpark函数冲突
        if test_days * one_day_length < max_possible_tests:
            dataset_length = test_days * one_day_length
        else:
            dataset_length = max_possible_tests
    else:
        raise ValueError("train mode: [{}] is not defined".format(train_mode))
    
    dataset = []
    for index in range(dataset_length):
        if train_mode == "train":
            actual_index = index
        elif train_mode == "test":
            actual_index = index + train_days * one_day_length + history_length
        
        try:
            data_x, data_y = slice_data(flow_data, history_length, actual_index, train_mode)
            dataset.append((data_x, data_y))
        except IndexError as e:
            print(f"Warning: Skipping index {actual_index} due to IndexError: {e}")
            continue
    
    return dataset

def gcn_layer(spark, graph, features_df, hidden_dim):
    """
    在Spark GraphFrames上实现GCN层
    :param spark: SparkSession
    :param graph: GraphFrame
    :param features_df: 节点特征DataFrame
    :param hidden_dim: 隐藏层维度
    :return: 经过GCN层处理后的特征
    """
    if not HAS_GRAPHFRAMES:
        # 如果没有GraphFrames，使用简化版实现
        return features_df
    
    # 1. 计算度矩阵的逆平方根
    # 计算每个节点的度
    degrees = graph.degrees
    
    # 计算度的逆平方根
    degrees_with_inv_sqrt = degrees.withColumn("inv_sqrt_degree", 
                                              when(col("degree") > 0, 
                                                   1.0 / sqrt(col("degree"))).otherwise(0.0))
    
    # 2. 将度信息与节点特征合并
    features_with_degree = features_df.join(degrees_with_inv_sqrt, 
                                           features_df.id == degrees_with_inv_sqrt.id, 
                                           "inner").drop(degrees_with_inv_sqrt.id)
    
    # 3. 执行图卷积操作
    # 通过消息传递机制聚合邻居信息
    # 发送消息：每个节点向其邻居发送特征乘以度的逆平方根
    messages = graph.edges.alias("e") \
        .join(features_with_degree.alias("src"), col("e.src") == col("src.id")) \
        .select(col("e.dst").alias("recipient"), 
                (col("src.feature") * col("src.inv_sqrt_degree")).alias("message"))
    
    # 接收消息：聚合来自邻居的消息
    aggregated = messages.groupBy("recipient") \
        .agg(sum("message").alias("aggregated_feature")) \
        .withColumnRenamed("recipient", "id")
    
    # 4. 应用度的逆平方根和线性变换
    result = aggregated.join(degrees_with_inv_sqrt, "id") \
        .withColumn("gcn_output", col("aggregated_feature") * col("inv_sqrt_degree"))
    
    return result

def gcn_prediction_model(spark, graph, train_dataset, test_dataset):
    """
    使用GCN进行交通流预测
    :param spark: SparkSession
    :param graph: GraphFrame图结构
    :param train_dataset: 训练数据集
    :param test_dataset: 测试数据集
    :return: 预测结果
    """
    # 简化的GCN预测模型
    # 在实际应用中，这里应该是一个更复杂的图神经网络模型
    predictions = []
    
    # 对于测试集中的每个样本进行预测
    for data_x, data_y in test_dataset:
        # 将数据转换为Spark DataFrame
        num_nodes = data_x.shape[0]
        
        # 简化的GCN操作：使用邻居节点的平均值作为预测
        if HAS_GRAPHFRAMES and graph:
            # 获取每个节点的邻居
            # 创建特征DataFrame
            features_list = [(i, float(np.mean(data_x[i]))) for i in range(num_nodes)]
            features_df = spark.createDataFrame(features_list, ["id", "feature"])
            
            # 应用GCN层
            try:
                gcn_output = gcn_layer(spark, graph, features_df, 16)
                # 收集结果并转换为numpy数组
                pred_values = gcn_output.select("id", "gcn_output").collect()
                pred = np.array([row.gcn_output for row in sorted(pred_values, key=lambda x: x.id)]).reshape(-1, 1)
            except Exception as e:
                print(f"GCN layer failed, using mean prediction: {e}")
                pred = np.mean(data_x, axis=1, keepdims=True)
        else:
            # 如果没有GraphFrames，使用简化版实现：节点历史数据的平均值
            pred = np.mean(data_x, axis=1, keepdims=True)
        
        predictions.append(pred)
    
    return np.array(predictions)

def main():
    """
    主函数
    """
    print("Setting up Spark session...")
    spark = setup_spark()
    
    print("Loading PeMS04 data...")
    graph, flow_data = load_pems04_data(spark)
    
    print(f"Flow data shape: {flow_data.shape}")
    
    print("Preprocessing flow data...")
    norm_base, normalized_flow_data = preprocess_flow_data(flow_data)
    
    # 参数设置
    history_length = 12  # 使用12个时间步的历史数据
    time_interval = 5    # 每5分钟一个数据点
    train_days = 45      # 使用30天数据训练
    test_days = 15        # 使用5天数据测试
    
    print("Creating training dataset...")
    train_dataset = create_dataset(normalized_flow_data, 
                                  [train_days, test_days], 
                                  time_interval, 
                                  history_length, 
                                  "train")
    
    print("Creating testing dataset...")
    test_dataset = create_dataset(normalized_flow_data, 
                                 [train_days, test_days], 
                                 time_interval, 
                                 history_length, 
                                 "test")
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    print("Running GCN prediction model...")
    predictions = gcn_prediction_model(spark, graph, train_dataset, test_dataset)
    
    print(f"Predictions shape: {predictions.shape}")
    print("Sample predictions:")
    print(predictions[:5].reshape(-1))  # 显示前5个节点的预测结果
    
    # 保存结果到HDF5文件
    try:
        with h5py.File('spark_gcn_pems04_result.h5', 'w') as f:
            f['predict'] = predictions
            f['target'] = np.array([data_y for _, data_y in test_dataset])
        print("Results saved to spark_gcn_pems04_result.h5")
    except Exception as e:
        print(f"Failed to save results: {e}")
    
    spark.stop()
    print("Spark session stopped.")

if __name__ == "__main__":
    main()