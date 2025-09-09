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
    # 尝试最简单的初始化方式
    try:
        spark = SparkSession.builder \
            .appName("TrafficPrediction") \
            .master("local[*]") \
            .getOrCreate()

            
    except:
        # 如果上面的方式失败，则使用基础配置
        spark = SparkSession.builder \
            .appName("TrafficPrediction") \
            .master("local[2]") \
            .config("spark.sql.warehouse.dir", "/tmp") \
            .config("spark.driver.host", "localhost") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .getOrCreate()
    
    return spark

def load_data(spark, csv_file, npz_file):
    """
    加载交通数据
    :param spark: SparkSession
    :param csv_file: CSV文件路径，包含节点间连接关系
    :param npz_file: NPZ文件路径，包含流量数据
    :return: graph_df, flow_data
    """
    # 加载图结构数据
    edges_df = spark.read.option("header", "true") \
        .option("inferSchema", "true") \
        .csv(csv_file)
    
    # 创建节点DataFrame
    nodes_list = set()
    edges_pandas = edges_df.toPandas()
    nodes_list.update(edges_pandas['from'].tolist())
    nodes_list.update(edges_pandas['to'].tolist())
    
    nodes_data = [(int(node),) for node in nodes_list]
    nodes_df = spark.createDataFrame(nodes_data, ["id"])
    
    # 重命名边的列以符合GraphFrames要求
    edges_renamed = edges_df.select(col("from").alias("src"), 
                                   col("to").alias("dst"), 
                                   col("cost").alias("weight"))
    
    # 创建GraphFrame
    graph = None
    if HAS_GRAPHFRAMES:
        try:
            graph = GraphFrame(nodes_df, edges_renamed)
        except Exception as e:
            print(f"Could not create GraphFrame: {e}")
            HAS_GRAPHFRAMES = False
    
    # 加载流量数据
    flow_data = np.load(npz_file)
    flow_array = flow_data['data'].transpose([1, 0, 2])[:, :, 0][:, :, np.newaxis]
    
    return graph, flow_array, edges_renamed

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

def simple_prediction_model(train_dataset):
    """
    简单的预测模型：基于历史平均值进行预测
    在Spark GraphFrames中，我们可以使用图结构特征来增强预测
    :param train_dataset: 训练数据集
    :return: 预测结果
    """
    # 这里实现一个简单的基于历史数据的预测模型
    # 实际应用中可以使用更复杂的图神经网络模型
    predictions = []
    for data_x, data_y in train_dataset:
        # 简单预测：使用历史数据的平均值作为预测值
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
    
    # 加载数据
    graph, flow_data, edges_df = load_data(spark, "PeMS_04/PeMS04.csv", "PeMS_04/PeMS04.npz")
    
    if graph and HAS_GRAPHFRAMES:
        print("Graph nodes: ", graph.vertices.count())
        print("Graph edges: ", graph.edges.count())
    else:
        print("Using basic Spark DataFrames")
        print("Edges: ", edges_df.count())
    
    # 预处理流量数据
    norm_base, norm_flow_data = preprocess_flow_data(flow_data)
    
    # 划分数据集
    time_interval = 5  # 5分钟
    history_length = 6  # 使用6个历史时间点
    divide_days = [45, 14]  # 训练45天，测试14天
    
    # 创建训练和测试数据集
    train_dataset = create_dataset(norm_flow_data, divide_days, time_interval, history_length, "train")
    test_dataset = create_dataset(norm_flow_data, divide_days, time_interval, history_length, "test")
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 简单预测 - 在实际应用中可以使用更复杂的图神经网络
    test_predictions = simple_prediction_model(test_dataset)
    
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
    result_file = "spark_gf_result.h5"
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