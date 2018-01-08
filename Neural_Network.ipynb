# Neural-Network

#行列計算ライブラリ
import numpy
#シグモイド関数の利用
import scipy.special
#配列の可視化
import matplotlib.pyplot
#描画の際、外部ウィンドウは開かない
%matplotlib inline

#ニューラルネットワーククラスの定義
class neuralNetwork:
    
    #ニューラルネットワークの初期化
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #入力層、隠れ層、出力層のノード数の設定
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        #リンクの重み行列
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        #学習率の設定
        self.lr = learningrate
        
        #活性化関数はシグモイド関数
        self.activation_function = lambda x : scipy.special.expit(x)
        
        pass
    
    #ニューラルネットワークの学習
    def train(self, inputs_list, targets_list):
        #入力リストを行列に変換
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        #隠れ層に入ってくる信号の計算
        hidden_inputs = numpy.dot(self.wih, inputs)
        #隠れ層で結合された信号を活性化関数により出力
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #出力層に入ってくる信号の計算
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #出力層で結合された信号を活性化関数により出力
        final_outputs = self.activation_function(final_inputs)
        
        #出力層の誤差の計算　（目標出力 - 最終出力）
        output_errors = targets - final_outputs
        #隠れ層の誤差の計算（出力層の誤差をリンクの重みに応じた割合で分配）
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        #隠れ層と出力層の間のリンクの重みの更新
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        #入力層と隠れ層の間のリンクの重みの更新
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass
    
    #ニューラルネットワークへの照会
    def query(self, inputs_list):
        #入力リストを行列に変換
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        #隠れ層に入ってくる信号の計算
        hidden_inputs = numpy.dot(self.wih, inputs)
        #隠れ層で結合された信号を活性化関数により出力
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #出力層に入ってくる信号の計算
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #出力層で結合された信号を活性化関数により出力
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    
    
#入力層、隠れ層、出力層のノード数
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

#学習率の設定
learning_rate = 0.3

#ニューラルネットワークのインスタンスの生成
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


#MNIST 訓練データCSVの読み込み、リスト化
training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#ニューラルネットワークの学習

#訓練データのすべてに対して実行
for record in training_data_list:
    #データを','で分割
    all_values = data_list[0].split(',')
    #scale input to range 0.01 to 1.00
    inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
    #target配列の作成（正解位置が 0.99 他は 0.01）
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    
    n.train(inputs, targets)
    
    pass
