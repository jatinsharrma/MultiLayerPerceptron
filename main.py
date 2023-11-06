from data import Data
from mlp import MLP
from plot import Plot
import matplotlib.pyplot as plt
from datetime import datetime

file_name = datetime.now().strftime("%Y%m%d%H%M%S")

# loading data
data = Data()
data.loadData("phishing_website.csv")
data.splitData()
Plot.plotData(data.data,data.heading,data.classes)

# working with MLP
mlp = MLP(data.no_feature,data.no_classes)

    #taining mlp
train_result = mlp.train(data.train)
mlp.save(file_name)

        # ploting result
Plot.plotResult(train_result)

    # testing mlp
test_result = mlp.test(data.test)
print("#------------------------------------------------------#")
print("#-----------------------Test Results-------------------#")
print("#------------------------------------------------------#")
print("Total Tests : ",test_result["total"])
print("Correct Predictions : ",test_result["correct"])
print("Accuracy :",test_result["accuracy"])
