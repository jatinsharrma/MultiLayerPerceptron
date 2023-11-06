import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

class Plot:
    @staticmethod
    def plotData(data,heading,classes):
        params = {'axes.titlesize':'8',
          'xtick.labelsize':'8',
          'ytick.labelsize':'8'}
        matplotlib.rcParams.update(params)

        df = pd.DataFrame(data,columns=heading)
        
        fig = df.hist(figsize=(20,10),alpha=0.4)
        plt.tight_layout()
        plt.savefig("plots/datapoints_1.png")

        for _class in classes:
            class_data = df[df['Result\n'] == _class]
            plt.scatter(class_data["having_IP_Address"], class_data["URL_Length"])
    
        plt.savefig("plots/datapoints_2.png")

    @staticmethod
    def plotResult(train_result):
        plt.plot(train_result["loss"])
        plt.title("Loss vs Epochs")
        plt.xlabel("Epoch Number")
        plt.ylabel("Loss Value")
        plt.savefig("plots/loss.png")

        plt.plot(train_result["accuracy"])
        plt.title("Accuracy vs Epochs")
        plt.xlabel("Epoch Number")
        plt.ylabel("Accuracy Value")
        plt.savefig("plots/accuracy.png")