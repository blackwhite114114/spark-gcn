import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

def setup_spark():
    """
    设置Spark会话
    """
    spark = SparkSession.builder \
        .appName("Simple GCN Traffic Prediction") \
        .master("local[*]") \
        .getOrCreate()
    
    return spark

def generate_test_data(spark):
    """
    生成测试数据：路网结构和交通流数据
    :param spark: SparkSession
    :return: edges_df, flow_data
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
    
    return edges_df, flow_data

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

def simple_gcn_layer(spark, edges_df, features):
    """
    简化版GCN层实现，基于Spark DataFrame操作
    :param spark: SparkSession
    :param edges_df: 边的DataFrame
    :param features: 节点特征 numpy array
    :return: 经过GCN层处理后的特征
    """
    # 将特征转换为Spark DataFrame
    num_nodes = features.shape[0]
    features_list = [(i, float(features[i])) for i in range(num_nodes)]
    features_df = spark.createDataFrame(features_list, ["id", "feature"])
    
    # 计算节点度数
    out_degrees = edges_df.groupBy("src").count().withColumnRenamed("src", "id").withColumnRenamed("count", "out_degree")
    in_degrees = edges_df.groupBy("dst").count().withColumnRenamed("dst", "id").withColumnRenamed("count", "in_degree")
    
    # 合并出度和入度
    degrees_df = out_degrees.join(in_degrees, "id", "outer") \
        .fillna(0) \
        .withColumn("degree", col("out_degree") + col("in_degree")) \
        .withColumn("inv_sqrt_degree", when(col("degree") > 0, 1.0 / sqrt(col("degree"))).otherwise(0.0))
    
    # 将度数信息与特征合并
    features_with_degree = features_df.join(degrees_df, "id", "left_outer").fillna(0)
    
    # 消息传递：每个节点向邻居发送特征信息
    # 通过边连接发送消息
    messages = edges_df.alias("e") \
        .join(features_with_degree.alias("src"), col("e.src") == col("src.id")) \
        .select(col("e.dst").alias("recipient"), 
                (col("src.feature") * col("src.inv_sqrt_degree")).alias("message"))
    
    # 聚合来自邻居的消息
    aggregated = messages.groupBy("recipient") \
        .agg(sum("message").alias("aggregated_feature")) \
        .withColumnRenamed("recipient", "id")
    
    # 应用度数归一化和激活函数
    result = aggregated.join(degrees_df, "id", "left_outer") \
        .withColumn("gcn_output", col("aggregated_feature") * col("inv_sqrt_degree")) \
        .select("id", "gcn_output")
    
    # 收集结果
    result_data = result.collect()
    output_features = np.array([row.gcn_output for row in sorted(result_data, key=lambda x: x.id)])
    
    return output_features.reshape(-1, 1)

def gcn_prediction_model(spark, edges_df, train_dataset, test_dataset):
    """
    使用GCN进行交通流预测
    :param spark: SparkSession
    :param edges_df: 路网结构DataFrame
    :param train_dataset: 训练数据集
    :param test_dataset: 测试数据集
    :return: 预测结果
    """
    predictions = []
    
    # 对于测试集中的每个样本进行预测
    for data_x, data_y in test_dataset:
        # 简化的GCN操作：使用邻居节点的平均值作为预测
        try:
            # 使用历史数据的均值作为输入特征
            features = np.mean(data_x, axis=1)
            # 应用GCN层
            gcn_output = simple_gcn_layer(spark, edges_df, features)
            pred = gcn_output
        except Exception as e:
            print(f"GCN layer failed, using mean prediction: {e}")
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
    
    # 生成测试数据
    edges_df, flow_data = generate_test_data(spark)
    
    print(f"Edges count: {edges_df.count()}")
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
    test_predictions = gcn_prediction_model(spark, edges_df, train_dataset, test_dataset)
    
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
    
    # 展示路网的一些基本分析
    try:
        # 计算节点的度数统计
        out_degrees = edges_df.groupBy("src").count()
        print("Graph out-degree statistics:")
        out_degrees.describe("count").show()
    except Exception as e:
        print(f"Graph analysis failed: {e}")
    
    spark.stop()

if __name__ == "__main__":
    main()