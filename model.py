import numpy as np
class MLP:
    def __init__(self,input=784,hidden=128,output=10,activation_type='relu',l2=0.0):
        self.input=input
        self.hidden=hidden
        self.output=output
        self.activation_type=activation_type
        self.l2=l2
        np.random.seed(42)
        self.W1=np.random.randn(self.input,self.hidden)*0.01
        self.b1=np.zeros((1,self.hidden))
        self.W2=np.random.randn(self.hidden,self.output)*0.01
        self.b2=np.zeros((1,self.output))
        self.cache={}
    def relu(self,z):
        return np.maximum(0,z)
    def relu_derivative(self,z):
        return (z>0).astype(float)
    def leaky_relu(self,z,alpha=0.01):
        return np.where(z>0,z,alpha*z)
    def leaky_relu_derivative(self,z,alpha=0.01):
        return np.where(z>0,1.0,alpha)
    def sigmoid(self,z):
        z=np.clip(z,-500,500)
        return 1/(1+np.exp(-z))
    def sigmoid_derivative(self,z):
        s=self.sigmoid(z)
        return s*(1-s)
    def gelu(self,z):
        return z*self.sigmoid(1.702*z)
    def gelu_derivative(self, z):
        sig = self.sigmoid(1.702*z)
        return sig+z*1.702*sig*(1-sig)
    def softmax(self,z):
        z0=z-np.max(z,axis=1,keepdims=True)
        exp_z0=np.exp(z0)
        return exp_z0/np.sum(exp_z0,axis=1,keepdims=True)
    def forward(self,X):
        self.cache['X']=X
        self.cache['Z1']=np.dot(X,self.W1)+self.b1
        if self.activation_type=='relu':
            self.cache['A1']=self.relu(self.cache['Z1'])
        elif self.activation_type=='leaky_relu':
            self.cache['A1']=self.leaky_relu(self.cache['Z1'])
        elif self.activation_type == 'sigmoid':
            self.cache['A1'] = self.sigmoid(self.cache['Z1'])
        elif self.activation_type=='gelu':
            self.cache['A1']=self.gelu(self.cache['Z1'])
        self.cache['Z2']=np.dot(self.cache['A1'],self.W2)+self.b2
        self.cache['A2']=self.softmax(self.cache['Z2'])
        return self.cache['A2']
    def backward(self,Y):
        m=Y.shape[0]
        Y1_hot=np.zeros((m,self.output))
        Y1_hot[range(m),Y]=1
        dZ2=self.cache['Z2']-Y1_hot
        dW2=np.dot(self.cache['A1'].T,dZ2)/m+self.l2*self.W2
        db2=np.sum(dZ2,axis=0,keepdims=True)/m
        dA1=np.dot(dZ2,self.W2.T)
        if self.activation_type=='relu':
            dZ1=dA1*self.relu_derivative(self.cache['Z1'])
        elif self.activation_type=='leaky_relu':
            dZ1=dA1*self.leaky_relu_derivative(self.cache['Z1'])
        elif self.activation_type=='sigmoid':
            dZ1=dA1*self.sigmoid_derivative(self.cache['Z1'])
        elif self.activation_type=='gelu':
            dZ1=dA1*self.gelu_derivative(self.cache['Z1'])
        dW1=np.dot(self.cache['X'].T,dZ1)/m+self.l2*self.W1
        db1=np.sum(dZ1,axis=0,keepdims=True)/m
        return{'dW1':dW1,'db1':db1,'dW2':dW2,'db2':db2}
    def compute_loss(self,Y_pred,Y_true):
        m=Y_true.shape[0]
        corect=-np.log(Y_pred[range(m),Y_true]+1e-8)
        data_loss=np.sum(corect)/m
        data_loss+=0.5*self.l2*(np.sum(self.W1**2)+np.sum(self.W2**2))
        return data_loss
if __name__ == '__main__':
    dummy_X = np.random.randn(64, 784)
    dummy_Y = np.random.randint(0, 10, size=(64,))
    print("正在初始化网络...")
    model = MLP(input=784, hidden=128, output=10, activation_type='relu', l2=0.01)
    print("\n测试前向传播...")
    preds = model.forward(dummy_X)
    print(f"预测输出形状: {preds.shape} (应该是 64, 10)")
    print(f"第一张图片的概率之和: {np.sum(preds[0]):.4f} (应该是 1.0)")
    print("\n测试损失计算...")
    loss = model.compute_loss(preds, dummy_Y)
    print(f"当前 Loss 值: {loss:.4f}")
    print("\n测试反向传播...")
    grads = model.backward(dummy_Y)
    print(f"dW1 形状: {grads['dW1'].shape} (应该是 784, 128)")
    print(f"db1 形状: {grads['db1'].shape} (应该是 1, 128)")
    print(f"dW2 形状: {grads['dW2'].shape} (应该是 128, 10)")
    print(f"db2 形状: {grads['db2'].shape} (应该是 1, 10)")
    print("\n恭喜！如果没报错，矩阵维度完美契合，反向传播大功告成！")