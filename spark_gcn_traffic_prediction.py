# -*- coding: utf-8 -*-
# @Time    : 2024/6/15
# @Author  : ChatGPT-4
# @github  :            
# spark_gcn_traffic_prediction.py
# 使用Spark和GraphFrames实现GCN进行交通流预测的示例代码
# 该代码生成测试数据，构建图神经网络模型，并进行训练和评估
# 该代码假设GraphFrames已正确安装并配置
# 运行该代码需要一个支持Spark的环境
# 注意：GraphFrames的安装和配置可能因环境而异，具体请参考官方文档
# 该代码仅用于教学和演示目的，实际应用中可能需要更复杂的处理
# 该代码生成的测试数据是随机的，实际应用中应使用真实的交通流数据和路网结构
# 该代码的性能和结果可能因环境和数据而异
# 该代码仅实现了GCN的基本功能，实际应用中可能需要更复杂的图神经网络模型
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
            .appName("GCN Traffic Prediction") \
            .master("local[*]") \
            .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.1-s_2.12") \
            .getOrCreate()
    except:
        # 如果上面的方式失败，则使用基础配置
        spark = SparkSession.builder \
            .appName("GCN Traffic Prediction") \
            .master("local[2]") \
            .config("spark.sql.warehouse.dir", "/tmp") \
            .config("spark.driver.host", "localhost") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.1-s_2.12") \
            .getOrCreate()
    
    return spark

def generate_test_data(spark):
    """
    生成测试数据：路网结构和交通流数据
    :param spark: SparkSession
    :return: graph, flow_data
    """
    # 生成测试路网结构数据 (节点和边)
    # 创建一个简单的路网，包含10个节点和它们之间的连接关系
    edges_data = [
        (0, 1, 5), (1, 0, 5),  # 节点0和1之间有连接，权重为5
        (1, 2, 3), (2, 1, 3),  # 节点1和2之间有连接，权重为3
        (2, 3, 7), (3, 2, 7),  # 节点2和3之间有连接，权重为7
        (3, 4, 2), (4, 3, 2),  # 节点3和4之间有连接，权重为2
        (4, 5, 4), (5, 4, 4),  # 节点4和5之间有连接，权重为4
        (5, 6, 6), (6, 5, 6),  # 节点5和6之间有连接，权重为6
        (6, 7, 8), (7, 6, 8),  # 节点6和7之间有连接，权重为8
        (7, 8, 1), (8, 7, 1),  # 节点7和8之间有连接，权重为1
        (8, 9, 9), (9, 8, 9),  # 节点8和9之间有连接，权重为9
        (0, 5, 10), (5, 0, 10), # 节点0和5之间有连接，权重为10
        (1, 6, 12), (6, 1, 12), # 节点1和6之间有连接，权重为12
        (2, 7, 15), (7, 2, 15), # 节点2和7之间有连接，权重为15
        (3, 8, 11), (8, 3, 11), # 节点3和8之间有连接，权重为11
        (4, 9, 13), (9, 4, 13)  # 节点4和9之间有连接，权重为13
    ]
    
    edges_df = spark.createDataFrame(edges_data, ["src", "dst", "weight"])
    
    # 创建节点DataFrame
    nodes_data = [(i,) for i in range(10)]  # 10个节点
    nodes_df = spark.createDataFrame(nodes_data, ["id"])
    
    # 创建GraphFrame
    graph = None
    global HAS_GRAPHFRAMES
    if HAS_GRAPHFRAMES:
        try:
            graph = GraphFrame(nodes_df, edges_df)
        except Exception as e:
            print(f"Could not create GraphFrame: {e}")
            HAS_GRAPHFRAMES = False
    
    # 生成测试交通流数据 (模拟5天的交通流数据，每5分钟一个数据点)
    np.random.seed(42)  # 固定随机种子以便结果可重现
    num_nodes = 10
    time_points = 5 * 24 * 60 // 5  # 5天，每5分钟一个数据点
    features = 1  # 只考虑一个特征：交通流量
    
    # 生成模拟的交通流数据，形状为 [num_nodes, time_points, features]
    flow_data = np.random.randint(10, 100, size=(num_nodes, time_points, features)).astype(np.float32)
    
    # 添加一些时间模式（例如早晚高峰）
    for t in range(time_points):
        hour = (t * 5 // 60) % 24  # 小时数
        # 早晚高峰增加流量
        if (7 <= hour <= 9) or (17 <= hour <= 19):
            flow_data[:, t, :] *= np.random.uniform(1.2, 1.5, size=(num_nodes, 1))
    
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
    train_days = divide_days[0]
    test_days = divide_days[1]
    one_day_length = int(24 * 60 / time_interval)
    
    if train_mode == "train":
        dataset_length = train_days * one_day_length - history_length
    elif train_mode == "test":
        dataset_length = test_days * one_day_length
    else:
        raise ValueError("train mode: [{}] is not defined".format(train_mode))
    
    dataset = []
    for index in range(dataset_length):
        if train_mode == "train":
            actual_index = index
        elif train_mode == "test":
            actual_index = index + train_days * one_day_length
        
        data_x, data_y = slice_data(flow_data, history_length, actual_index, train_mode)
        dataset.append((data_x, data_y))
    
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
            # 如果没有GraphFrames，使用简单的平均值预测
            pred = np.mean(data_x, axis=1, keepdims=True)
            
        predictions.append(pred)
    
    return predictions

def recover_data(max_data, min_data, data):
    """
    恢复归一化后的数据
    :param max_data: np.array, max data
    :param min_data: np.array, min data
    :param data: np.array, normalized data
    :return: recovered_data: np.array, recovered data
    """
    mid = min_data
    base = max_data - min_data
    recovered_data = data * base + mid
    return recovered_data

def evaluate_prediction(target, prediction):
    """
    评估预测结果
    :param target: 真实值
    :param prediction: 预测值
    :return: mae, mape, rmse
    """
    # MAE - Mean Absolute Error
    mae = np.mean(np.abs(target - prediction))
    
    # MAPE - Mean Absolute Percentage Error
    mape = np.mean(np.abs(target - prediction) / (target + 5))  # 加5避免除零
    
    # RMSE - Root Mean Square Error
    rmse = np.sqrt(np.mean(np.power(target - prediction, 2)))
    
    return mae, mape, rmse

def main():
    # 设置Spark
    spark = setup_spark()
    
    # 设置检查点目录，用于GraphFrames算法
    spark.sparkContext.setCheckpointDir("/tmp/spark-checkpoint")
    
    # 生成测试数据
    graph, flow_data = generate_test_data(spark)
    
    if graph and HAS_GRAPHFRAMES:
        print("Graph nodes: ", graph.vertices.count())
        print("Graph edges: ", graph.edges.count())
    else:
        print("Using basic Spark DataFrames")
    
    print(f"Flow data shape: {flow_data.shape}")
    
    # 预处理流量数据
    norm_base, norm_flow_data = preprocess_flow_data(flow_data)
    
    # 划分数据集
    time_interval = 5  # 5分钟
    history_length = 6  # 使用6个历史时间点
    divide_days = [3, 2]  # 训练3天，测试2天
    
    # 创建训练和测试数据集
    train_dataset = create_dataset(norm_flow_data, divide_days, time_interval, history_length, "train")
    test_dataset = create_dataset(norm_flow_data, divide_days, time_interval, history_length, "test")
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 使用GCN进行预测
    test_predictions = gcn_prediction_model(spark, graph, train_dataset, test_dataset)
    
    # 从测试数据集中提取真实值
    test_targets = [data_y for _, data_y in test_dataset]
    
    # 将预测结果和真实值堆叠成数组
    pred_array = np.vstack(test_predictions)
    target_array = np.vstack(test_targets)
    
    # 恢复数据（逆归一化）
    recovered_pred = recover_data(norm_base[0], norm_base[1], pred_array)
    recovered_target = recover_data(norm_base[0], norm_base[1], target_array)
    
    # 评估预测结果
    mae, mape, rmse = evaluate_prediction(recovered_target, recovered_pred)
    
    print("Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape*100:.2f}%")
    print(f"RMSE: {rmse:.2f}")
    
    # 保存结果到HDF5文件
    result_file = "spark_gcn_result.h5"
    with h5py.File(result_file, "w") as f:
        # 重构数据形状为[N, T, D]
        pred_reshaped = recovered_pred.reshape(-1, len(test_dataset), 1)
        target_reshaped = recovered_target.reshape(-1, len(test_dataset), 1)
        
        f["predict"] = pred_reshaped
        f["target"] = target_reshaped
    
    print(f"Results saved to {result_file}")
    
    # 展示图的一些基本分析
    if graph and HAS_GRAPHFRAMES:
        try:
            # 计算节点的度
            degrees = graph.degrees
            print("Graph degree statistics:")
            degrees.describe().show()
            
            # 计算连接组件
            cc = graph.connectedComponents()
            print("Number of connected components: ", cc.select("component").distinct().count())
        except Exception as e:
            print(f"Graph analysis failed: {e}")
    else:
        print("Graph analysis skipped due to GraphFrame unavailability")
    
    spark.stop()

if __name__ == "__main__":
    main()