import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from data_loader import load_fashion_mnist
from model import MLP
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
def load_best_model():
    if not os.path.exists('weights/best_model.npz'):
        raise FileNotFoundError("找不到权重文件！请确保你已经成功运行了训练脚本。")
    model = MLP(input=784, hidden=128, output=10, activation_type='gelu')
    data = np.load('weights/best_model.npz')
    model.W1 = data['W1']
    model.b1 = data['b1']
    model.W2 = data['W2']
    model.b2 = data['b2']
    print("成功加载最优模型权重！")
    return model
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title('Confusion Matrix on Test Set')
    plt.tight_layout()
    plt.show()
def visualize_weights(model):
    W1 = model.W1
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("Visualization of First Hidden Layer Weights", fontsize=16)
    for i, ax in enumerate(axes.flatten()):
        weight_vector = W1[:, i]
        weight_image = weight_vector.reshape(28, 28)
        ax.imshow(weight_image, cmap='coolwarm')
        ax.axis('off')
        ax.set_title(f'Neuron {i + 1}')
    plt.tight_layout()
    plt.show()
def error_analysis(X_test, y_true, y_pred):
    errors = np.where(y_pred != y_true)[0]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Error Analysis: Misclassified Examples", fontsize=16)
    np.random.seed(42)
    sample_errors = np.random.choice(errors, 6, replace=False)
    for i, ax in enumerate(axes.flatten()):
        idx = sample_errors[i]
        image = X_test[idx].reshape(28, 28)
        true_label = CLASS_NAMES[y_true[idx]]
        pred_label = CLASS_NAMES[y_pred[idx]]
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', color='red')
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    _, _, _, _, X_test, y_test = load_fashion_mnist()
    best_model = load_best_model()
    print("\n正在测试集上进行最终评估...")
    test_probs = best_model.forward(X_test)
    y_pred = np.argmax(test_probs, axis=1)
    test_acc = np.mean(y_pred == y_test)
    print(f"\n最终测试集准确率 (Test Accuracy): {test_acc * 100:.2f}%")
    print("\n生成混淆矩阵...")
    plot_confusion_matrix(y_test, y_pred)
    print("\n生成权重可视化...")
    visualize_weights(best_model)
    print("\n生成错例分析图...")
    error_analysis(X_test, y_test, y_pred)