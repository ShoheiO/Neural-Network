#入力層、隠れ層、出力層のノード数
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

#学習率の設定
learning_rate = 0.3

#ニューラルネットワークのインスタンスの生成
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

n.query([1.0, 0.5, -1.5])
