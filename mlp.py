import numpy as np
from copy import deepcopy

class Sigmoid:
    """
    The sigmoid function.
    """
    def main(var:np.array):
        """
        Calculate the sigmoid of var.
        Parameters:
            var (np.array) - input value to calculate sigmoid for
        Returns:
            np.array - result of applying sigmoid function on var
        """
        return 1. / ( 1. + np.exp(-var) )
    
    def deriv(var:np.array):
        """
        Calculate the derivative of the sigmoid function at var.
        Parameters:
            var (np.array) - input value to find derivative at
        Returns:
            np.array - derivative of sigmoid function at var
        """
        return var * (1 - var)
    
class MSE:
    """
    Cost Function - Mean Squared Error
    """
    
    def main(D:np.array,Y:np.array):
        """
        Evaluate cost using log loss
        Parameters:
            D (np.array) - actual values
            Y (np.array) - predicted probabilities
        Returns:
            float - cost
        """
        return 0.5 * np.sum(np.power(D - Y, 2))


class MLP:
    """   Multi Layer perceptron    """
    def __init__(self,no_inputs, no_outputs,hidden=[3]) -> None:
        self.no_inp = no_inputs
        self.no_opt = no_outputs
        self.hid = hidden
        self.layers = [no_inputs] + hidden + [no_outputs]

        self.wt = self.initializeWeights()
        self.b = self.initializeBias()

    def initializeWeights(self):
        """
        Initializing weight matrix for MLP.
        Returns:
            list of ndarray - initialized weights
        """
        wt = []
        for i in range(len(self.layers)-1):
            wt.append(np.random.rand(self.layers[i],self.layers[i+1])/10)
        return wt
    
    def initializeBias(self):   
        """
        Initializing bias matrix for MLP.
        Returns:
            list of ndarray - initialized biases
        """
        wt = []
        for i in range(len(self.layers)-1):
            wt.append(np.zeros((self.layers[i+1],1)))
        return wt
    
    def train(self,train:tuple,eta=0.8,epoch=10):
        """
        Train MLP. There are 2 stages in this
            1. Forward propogation
            2. Backward propogation (uses Error back propogation)
        Parameters:
            train (list of ndarray) - training data and labels
            eta (float) - learning rate
            epoch (int) - total number of epoch
        Return:
            dict - dictionary of loss and accuracy of each epoch
        """
        no_input = train[0].shape[0]
        loss = []
        accuracy = []

        for i in range(epoch):
            cost = 0
            correct = 0
            for i in range(no_input):
                p = train[0][i].reshape(self.no_inp,1)
                d = train[1][i].reshape(self.no_opt,1)

                # performing feed forward
                outputs_per_layer:list = self.predict(p)
                outputs_per_layer.insert(0,p)

                # performing backpropogation

                error = 0
                for l in range(-1,-len(self.layers),-1):
                    if l == -1:         # for output layer
                        error = d - outputs_per_layer[-1]
                    else:               # for hidden layers
                        error = np.matmul(self.wt[l+1],error)
                    local_gradiant = error * Sigmoid.deriv(outputs_per_layer[l])
                    change_in_wt = eta * np.matmul(outputs_per_layer[l-1],local_gradiant.T)

                    self.wt[l] = self.wt[l] + change_in_wt
                    self.b[l] = self.b[l] + np.sum(local_gradiant)

                cost += MSE.main(d,outputs_per_layer[-1])
                if self.isCorrect(d,outputs_per_layer[-1]):
                    correct+=1
            loss.append(cost/no_input)
            accuracy = self.accuracy(no_input,correct)
            print("Loss= "+ str(loss[-1]),end=" , ")
            print("Accuracy= " + str(accuracy))
        return {"loss":loss,
                "accuracy":accuracy}

    def accuracy(self,total:int,correct:int):
        """
        Calculate accuracy
        Parameters:
            total (int) - total number of sample points
            correct (int) - correctly classified sample points
        Rerturn :
            flot - percentage of the correct.
        """
        return correct/total
    
    def isCorrect(self,D,Y):
        """
        Check if D matches with Y
        Parameters:
            D (np.array) - actual values
            Y (np.array) - predicted values
        Returns:
            bool - True if correct else False
        """
        desired = 0
        predicted = 0

        for i in range(D.shape[0]):
            if D[i,0] == 1:
                desired = i
                
        max_y = Y[0,0]
        predicted = 0

        for i in range(Y.shape[0]):
            if Y[i,0] > max_y:
                max_y = Y[i,0]
                predicted = i

        if desired == predicted:
            return True
        else:
            return False

    
    def predict(self,pattern:np.array):
        """
        Predicts output using trained model. This is also the feed forward
        Parameters :
            pattern (np.array) - input to be classified
        Return:
            list of ndarray - output of each layer.
        """
        x = deepcopy(pattern)
        output = []
        for i in range(len(self.layers)-1):
            v = np.matmul(self.wt[i].T,x) + self.b[i]
            x = Sigmoid.main(v)
            output.append(x)
        return output
    
    def test(self,test:tuple):
        """
        Testing MLP with test data which contain the test points at index 0 and desired output at index 1
        Parameters:
            test (tuple of ndarray) - tuple containing test data and desired output
        Return:
            dict - dictionary of result
        """
        total_input = test[0].shape[0]

        correct = 0
        for p in range(total_input):
            y = test[0][p].reshape(test[0][p].shape[0],1)
            y = self.predict(y)[-1]
            d = test[1][p].reshape(test[1][p].shape[0],1)

            if self.isCorrect(d,y):
                correct += 1

        return {"correct":correct,
                "total":total_input,
                "accuracy":self.accuracy(total_input,correct)*100} 

    def save(self,file_name='model'):
        """
        Saves weights and biases into a file.
        """
        def write(type:str,arrays:list):
            with open(f"models/{file_name}","a") as model:
                for i in range(len(arrays)):
                    model.write(f"{type} Vector for layer :{i+1}\n")
                    for j in arrays[i]:
                        model.write(",".join([str(x) for x in j])+"\n")
        write("@Weight",self.wt)
        write("$Bias",self.b)

if __name__ == "__main__":
    from data import Data
    data = Data()
    data.loadData("phishing_website.csv")
    data.splitData()
    mlp = MLP(data.no_feature,data.no_classes)
    mlp.train(data.train)
    mlp.test(data.test)