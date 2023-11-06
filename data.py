import numpy as np

class Data:
    def __init__(self):
        self.data = []
        self.train = None
        self.test = None
        self.no_feature = 0
        self.no_classes = 0
        self.heading = None
        self.classes = []
    
    def loadData(self,path):
        data = []
        with open(path,"r") as file:
            file = file.readlines()
            self.heading = np.array(file[0].split(","))
            for line in file[1:]:
                split_line = np.array(line.split(","))
                split_line = split_line.astype(np.float32)
                data.append(split_line)
        self.data = np.asarray(data)
        total_columns = len(self.data[0])
        self.no_feature = total_columns-1
        self.classes = np.unique(self.data[:,total_columns-1])
        self.no_classes = len(self.classes)

    def splitData(self,ratio=0.8):
        train_row_count = int(len(self.data)*ratio)
        np.random.shuffle(self.data)
        self.train = (self.data[:train_row_count,:self.no_feature], 
                      self.oneHotencoding(self.data[:train_row_count,self.no_feature])[0])
        self.test = (self.data[train_row_count:,:self.no_feature] , 
                     self.oneHotencoding(self.data[train_row_count:,self.no_feature])[0])

        return self.train, self.test
    
    def oneHotencoding(self,array:np.array):
        unique_elements = np.unique(array)
        array_n = np.zeros((len(array),len(unique_elements)))
        for i in range(len(unique_elements)):
            array_n[np.where(array==unique_elements[i]),i] = 1
        return array_n,unique_elements
    
    def __str__(self) -> str:
        return f"No. of features : {self.no_feature} \nNo. of unique classes : {self.no_classes} \nHeadings : {np.array_str(self.heading)}"
    

if __name__ == "__main__":
    d = Data()
    d.loadData("phishing_website.csv")
    d.splitData()