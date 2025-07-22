import numpy as np
import os


class LoadData(object):

    def __init__(self, path, sub_dataset,  hidden_factor=50):
        self.path = path + sub_dataset
        self.trainfile = self.path + "RGPRecA_train.libfm"
        self.testfile = self.path + "RGPRecA_test.libfm"
        self.features_M_dev = self.map_features_dev()
        self.features_M_task = self.map_features_task()
        self.Train_data, self.Test_data = self.construct_data()


    def map_features_dev(self):
        self.features_dev = {}
        self.read_features_dev(self.trainfile)
        self.read_features_dev(self.testfile)
        
        print("features_M_dev:", len(self.features_dev))
        return len(self.features_dev)

    def read_features_dev(self, file):
        f = open(file)
        line = f.readline()
        i = len(self.features_dev)
        while line:
            items = line.strip().split(' ')
            if items[2] not in self.features_dev:
                self.features_dev[ items[2] ] = i
                i = i + 1
            for item in items[4:8]:
                if item not in self.features_dev:
                    self.features_dev[ item ] = i
                    i = i + 1
            line = f.readline()
        f.close()
        
    def map_features_task(self):
        self.features_task = {}
        self.read_features_task(self.trainfile)
        self.read_features_task(self.testfile)
        
        print("features_M_task:", len(self.features_task))

        return len(self.features_task)

    def read_features_task(self, file):  # read a feature file
        f = open(file)
        line = f.readline()
        i = len(self.features_task)
        while line:
            items = line.strip().split(' ')
            if items[3] not in self.features_task:
                self.features_task[ items[3] ] = i
                i = i + 1
            for item in items[8:]:
                if item not in self.features_task:
                    self.features_task[ item ] = i
                    i = i + 1
            line = f.readline()
        f.close()
        
    def construct_data(self):
        X_U, X_I, Y1 , Y2 = self.read_data(self.trainfile)
        Train_data = self.construct_dataset(X_U, X_I, Y1, Y2)
        # print("# number of training:" , len(Y1))
        
        X_U, X_I, Y1 , Y2 = self.read_data(self.testfile)
        Test_data = self.construct_dataset(X_U, X_I, Y1, Y2)
        # print("# number of test:", len(Y1))

        return Train_data, Test_data

    def read_data(self, file):
        f = open(file)
        X_U = []
        X_I = []
        Y1 = []
        Y2 = []
        line = f.readline()
        while line:
            items = line.strip().split(' ')
            Y1.append(1.0 * float(items[0]))
            Y2.append(1.0 * float(items[1]))
            if float(items[0]) > 0:
                v = items[0]
            else:
                v = 0.0

            X_U.append([ self.features_dev[items[2]],
                          self.features_dev[items[4]],
                          self.features_dev[items[5]],
                          self.features_dev[items[6]],
                          self.features_dev[items[7]]])
            
            X_I.append([ self.features_task[items[3]],
                         self.features_task[items[8]],
                         self.features_task[items[9]],
                         self.features_task[items[10]],
                         self.features_task[items[11]],
                         self.features_task[items[12]] ])
            line = f.readline()
        f.close()
        return X_U, X_I, Y1, Y2

    def construct_dataset(self, X_U, X_I, Y1, Y2):
        Data_Dic = {}
        X_U_lens = [len(line) for line in X_U]
        X_I_lens = [len(line) for line in X_I]
        
        indexs_U = np.argsort(X_U_lens)
        indexs_I = np.argsort(X_I_lens)

        Data_Dic['Y1'] = [ Y1[i] for i in indexs_U]
        #print(Data_Dic['Y1'][0])
        Data_Dic['Y2'] = [ Y2[i] for i in indexs_U]
        Data_Dic['X_U'] = [ X_U[i] for i in indexs_U]
        #print(Data_Dic['X_U'][0])
        Data_Dic['X_I'] = [ X_I[i] for i in indexs_I]
        return Data_Dic
    
