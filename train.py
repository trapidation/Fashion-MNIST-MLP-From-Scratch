import numpy as np
import matplotlib.pyplot as plt
import os
from data_loader import load_fashion_mnist
from model import MLP
def compute_accuracy(Y_pred_probs, Y_true):
    predictions = np.argmax(Y_pred_probs, axis=1)
    accuracy = np.mean(predictions == Y_true)
    return accuracy
def train_model(model, X_train, y_train, X_val, y_val,
                epochs=15, batch_size=128, learning_rate=0.1, lr_decay=0.95):
    m = X_train.shape[0]
    history = {'train_loss': [], 'val_loss': [],'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    best_weights = {}
    print(f"开始训练... 模型隐藏层大小: {model.hidden}, 激活函数: {model.activation_type}")
    for epoch in range(epochs):
        permutation = np.random.permutation(m)
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]
        for i in range(0, m, batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]
            preds = model.forward(X_batch)
            loss = model.compute_loss(preds, y_batch)
            grads = model.backward(y_batch)
            model.W1 -= learning_rate * grads['dW1']
            model.b1 -= learning_rate * grads['db1']
            model.W2 -= learning_rate * grads['dW2']
            model.b2 -= learning_rate * grads['db2']
        learning_rate = learning_rate * lr_decay
        train_preds = model.forward(X_train[:5000])
        train_loss = model.compute_loss(train_preds, y_train[:5000])
        train_acc = compute_accuracy(train_preds, y_train[:5000])
        val_preds = model.forward(X_val)
        val_loss = model.compute_loss(val_preds, y_val)
        val_acc = compute_accuracy(val_preds, y_val)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch + 1:02d}/{epochs} | "
              f"LR: {learning_rate:.4f} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc * 100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc * 100:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  --> 发现更好的模型！验证集准确率提升至 {best_val_acc * 100:.2f}%，已暂存权重。")
            best_weights = {
                'W1': model.W1.copy(), 'b1': model.b1.copy(),
                'W2': model.W2.copy(), 'b2': model.b2.copy()
            }
    if not os.path.exists('weights'):
        os.makedirs('weights')
    np.savez('weights/best_model.npz', **best_weights)
    print(f"\n训练完成！最佳验证集准确率: {best_val_acc * 100:.2f}%，权重已保存到 weights/best_model.npz")
    return history
def plot_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
# ==========================================
# 主程序：大满贯版 (一次性生成 3 个实验的所有图表)
# ==========================================
if __name__ == '__main__':
    # 1. 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_fashion_mnist()

    # ==========================================
    # 实验一：训练最优模型，生成作业必需的 Loss/Acc 曲线图
    # ==========================================
    print("\n" + "=" * 50)
    print("🚀 实验一：训练最优模型 (固定 ReLU, 128隐藏层)")
    print("=" * 50)

    best_model = MLP(input=784, hidden=128, output=10,
                     activation_type='relu', l2=0.001)

    best_history = train_model(best_model, X_train, y_train, X_val, y_val,
                               epochs=30, batch_size=64, learning_rate=0.1, lr_decay=0.95)

    print("\n[注意] 正在弹出第一张图(Loss/Acc曲线)，请查看后【关闭窗口】，程序才会继续执行后续实验！")
    plot_history(best_history)

    # ==========================================
    # 实验二：基本要求 - 4种激活函数 (Activation) 对比
    # ==========================================
    print("\n" + "=" * 50)
    print("🚀 实验二：对比 4 种激活函数对性能的影响 (固定 128隐藏层)")
    print("=" * 50)

    activations_to_test = ['sigmoid', 'relu', 'leaky_relu', 'gelu']
    act_histories = {}

    for act in activations_to_test:
        print(f"\n---> 正在测试激活函数: {act.upper()}")
        test_model = MLP(input=784, hidden=128, output=10,
                         activation_type=act, l2=0.001)

        history = train_model(test_model, X_train, y_train, X_val, y_val,
                              epochs=15, batch_size=64, learning_rate=0.1, lr_decay=0.95)
        act_histories[act] = history

    # 画出激活函数对比图
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, 16)
    styles = {
        'sigmoid': {'color': 'gray', 'marker': 'v'},
        'relu': {'color': 'blue', 'marker': 'o'},
        'leaky_relu': {'color': 'green', 'marker': 's'},
        'gelu': {'color': 'red', 'marker': '*'}
    }

    for act in activations_to_test:
        plt.plot(epochs_range, act_histories[act]['val_acc'],
                 label=f'{act.upper()} Val Acc',
                 color=styles[act]['color'], marker=styles[act]['marker'], linewidth=2)

    plt.title('Validation Accuracy Comparison (4 Activations)')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    print("\n[注意] 正在弹出第二张图(激活函数对比)，请查看后【关闭窗口】，程序才会继续执行最后一步！")
    plt.show()

    # ==========================================
    # 实验三：加分项 - 隐藏层大小 (Hidden Dimension) 对比
    # ==========================================
    print("\n" + "=" * 50)
    print("🚀 实验三：对比不同隐藏层大小对性能的影响 (固定 ReLU)")
    print("=" * 50)

    hidden_dims_to_test = [32, 64, 128, 256]
    dim_histories = {}

    for h_dim in hidden_dims_to_test:
        print(f"\n---> 正在测试隐藏层大小: {h_dim}")
        test_model = MLP(input=784, hidden=h_dim, output=10,
                         activation_type='relu', l2=0.001)

        history = train_model(test_model, X_train, y_train, X_val, y_val,
                              epochs=15, batch_size=64, learning_rate=0.1, lr_decay=0.95)
        dim_histories[h_dim] = history

    # 画出隐藏层对比图
    plt.figure(figsize=(10, 6))
    colors = ['gray', 'blue', 'green', 'red']
    markers = ['v', 'o', 's', '*']

    for i, h_dim in enumerate(hidden_dims_to_test):
        plt.plot(epochs_range, dim_histories[h_dim]['val_acc'],
                 label=f'Hidden Dim: {h_dim}',
                 color=colors[i], marker=markers[i], linewidth=2)

    plt.title('Validation Accuracy Comparison (Different Hidden Dimensions)')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    print("\n[注意] 正在弹出最后一张图(隐藏层对比)，查看后关闭即可。")
    plt.show()
    # ==========================================
    # 实验四：加分项 - L2 正则化强度 (L2 Regularization) 对比
    # ==========================================
    print("\n" + "=" * 50)
    print("🚀 实验四：对比不同 L2 正则化强度对性能的影响 (固定 ReLU, 128隐藏层)")
    print("=" * 50)

    # 我们测试 4 种强度：0.0(不加正则化), 0.001(轻度), 0.01(中度), 0.1(重度)
    l2_regs_to_test = [0.0, 0.001, 0.01, 0.1]
    reg_histories = {}

    for reg in l2_regs_to_test:
        print(f"\n---> 正在测试 L2 正则化强度: {reg}")
        # 固定使用 relu 和 128，只改变 l2_reg
        test_model = MLP(input=784, hidden=128, output=10,
                         activation_type='relu', l2=reg)

        # 同样跑 15 圈对比趋势
        history = train_model(test_model, X_train, y_train, X_val, y_val,
                              epochs=15, batch_size=64, learning_rate=0.1, lr_decay=0.95)
        reg_histories[reg] = history

    # 画出正则化对比图
    plt.figure(figsize=(10, 6))
    colors = ['gray', 'blue', 'green', 'red']
    markers = ['v', 'o', 's', '*']
    epochs_range = range(1, 16)

    for i, reg in enumerate(l2_regs_to_test):
        # 注意：画正则化图时，画 Validation Accuracy 最能说明问题
        plt.plot(epochs_range, reg_histories[reg]['val_acc'],
                 label=f'L2 Reg: {reg}',
                 color=colors[i], marker=markers[i], linewidth=2)

    plt.title('Validation Accuracy Comparison (Different L2 Regularization)')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    print("\n[注意] 正在弹出正则化对比图，查看后关闭即可。")
    plt.show()
    print("\n" + "=" * 50)
    print("🚀 实验五：对比不同初始学习率对性能的影响 (固定 ReLU, 128隐藏层)")
    print("=" * 50)

    # 我们测试 4 种学习率：0.01(太慢), 0.05(适中), 0.1(最优), 0.5(太大)
    lrs_to_test = [0.01, 0.05, 0.1, 0.5]
    lr_histories = {}

    for lr in lrs_to_test:
        print(f"\n---> 正在测试初始学习率: {lr}")
        test_model = MLP(input=784, hidden=128, output=10,
                         activation_type='relu', l2=0.001)

        # 注意：这里的 learning_rate 换成了循环里的 lr
        history = train_model(test_model, X_train, y_train, X_val, y_val,
                              epochs=15, batch_size=64, learning_rate=lr, lr_decay=0.95)
        lr_histories[lr] = history

    # 画出学习率对比图
    plt.figure(figsize=(10, 6))
    colors = ['gray', 'blue', 'green', 'red']
    markers = ['v', 'o', 's', '*']
    epochs_range = range(1, 16)

    for i, lr in enumerate(lrs_to_test):
        plt.plot(epochs_range, lr_histories[lr]['val_acc'],
                 label=f'Learning Rate: {lr}',
                 color=colors[i], marker=markers[i], linewidth=2)

    plt.title('Validation Accuracy Comparison (Different Learning Rates)')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("\n🎉 所有超参数查找实验大满贯完成！！")
    print("\n🎉 所有实验运行完毕！三张王牌图表已全部保存在项目文件夹中！")
