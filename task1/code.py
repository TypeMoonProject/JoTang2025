import numpy as np

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU激活函数"""
    return np.maximum(0, x)

class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重和偏置 (通常使用随机值)
        # 为了简化，这里使用固定的示例值或随机初始化
        # W_h: hidden_size x input_size
        # b_h: hidden_size x 1
        # W_o: output_size x hidden_size
        # b_o: output_size x 1
        
        # np.random.seed(42) # 为了可复现性
        self.W_h = np.random.randn(hidden_size, input_size) * 0.1 
        self.b_h = np.zeros((hidden_size, 1))
        self.W_o = np.random.randn(output_size, hidden_size) * 0.1
        self.b_o = np.zeros((output_size, 1))
        
        print("Initialized W_h:\n", self.W_h)
        print("Initialized b_h:\n", self.b_h)
        print("Initialized W_o:\n", self.W_o)
        print("Initialized b_o:\n", self.b_o)

    def forward(self, X):
        """
        MLP的前向传播
        X: 输入数据，形状为 (input_size, 1) 或 (input_size, num_samples)
        """
        # 确保 X 是一个列向量 (input_size, 1) 如果只是单个样本
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # 输入层到隐藏层
        # Z_h = W_h * X + b_h
        Z_h = np.dot(self.W_h, X) + self.b_h  # 线性变换
        A_h = relu(Z_h)                      # 应用ReLU激活函数
        
        print("\n--- Hidden Layer ---")
        print("Z_h (Linear Output):\n", Z_h)
        print("A_h (After ReLU):\n", A_h)
        
        # 隐藏层到输出层
        # Z_o = W_o * A_h + b_o
        Z_o = np.dot(self.W_o, A_h) + self.b_o # 线性变换
        A_o = sigmoid(Z_o)                   # 应用Sigmoid激活函数 (预测概率)
        
        print("\n--- Output Layer ---")
        print("Z_o (Linear Output):\n", Z_o)
        print("A_o (Prediction, After Sigmoid):\n", A_o)
        
        return A_o

# 示例使用
if __name__ == '__main__':
    # 定义MLP结构: 输入层2个节点, 隐藏层3个节点, 输出层1个节点
    input_features = 2
    hidden_nodes = 3
    output_nodes = 1
    
    mlp = SimpleMLP(input_size=input_features, 
                    hidden_size=hidden_nodes, 
                    output_size=output_nodes)
    
    # 假设一个输入样本
    sample_X = np.array([0.1, 0.2]) 
    print("\nInput X:\n", sample_X.reshape(-1,1))
    
    # 执行前向传播
    prediction = mlp.forward(sample_X)
    
    print(f"\nFinal Prediction for input {sample_X}: {prediction[0,0]:.4f}")

    # 另一个输入样本
    sample_X_2 = np.array([0.8, 0.9])
    print("\nInput X:\n", sample_X_2.reshape(-1,1))
    prediction_2 = mlp.forward(sample_X_2)
    print(f"\nFinal Prediction for input {sample_X_2}: {prediction_2[0,0]:.4f}")
