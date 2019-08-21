from imageai.Prediction import ImagePrediction
import os

#PATH=$PATH:"c:\\Users\\Anthony2013"

#execution_path = os.getcwd()
execution_path="c:\\Users\\Anthony2013"
prediction = ImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath( execution_path + "\\resnet50_weights_tf_dim_ordering_tf_kernels.h5")
prediction.loadModel()

predictions,percentage_probabilities=prediction.predictImage("c:\\Users\\Anthony2013\\Documents\python3_start-master\\sample.jpg", result_count=5)

for index in range(len(predictions)):
    print(predictions[index] , " : " , percentage_probabilities[index])
