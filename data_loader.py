import numpy as np
import gzip
import os
def load_fashion_mnist(data_path='data'):
    train_images_path = os.path.join(data_path, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_path, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_path, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_path, 't10k-labels-idx1-ubyte.gz')
    def read_images(file_path):
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28 * 28).astype(np.float32) / 255.0
        return data
    def read_labels(file_path):
        with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels.astype(np.int64)
    print("正在加载数据，请稍候...")
    X_train_full = read_images(train_images_path)
    y_train_full = read_labels(train_labels_path)
    X_test = read_images(test_images_path)
    y_test = read_labels(test_labels_path)
    split_index = 50000
    X_train = X_train_full[:split_index]
    y_train = y_train_full[:split_index]
    X_val = X_train_full[split_index:]
    y_val = y_train_full[split_index:]
    print("数据加载成功！")
    return X_train, y_train, X_val, y_val, X_test, y_test
if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_fashion_mnist()
    print("-------------------------")
    print(f"训练集图像矩阵形状: {X_train.shape}")  
    print(f"训练集标签矩阵形状: {y_train.shape}")  
    print(f"验证集图像矩阵形状: {X_val.shape}")  
    print(f"验证集标签矩阵形状: {y_val.shape}")  
    print(f"测试集图像矩阵形状: {X_test.shape}")  
    print(f"测试集标签矩阵形状: {y_test.shape}")  
    print("-------------------------")
    print("检查归一化是否成功 (打印前几个像素值):", X_train[0][150:155])