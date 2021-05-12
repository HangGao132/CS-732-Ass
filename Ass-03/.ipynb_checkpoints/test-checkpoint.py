import pandas as pd
import os
import math


class MyBayesCLF():
    def __init__(self):
        self.__data_Class = {}
        self.__each_Class_Words = {}
        self.__each_Class_Word_Num = {}
        self.__p = {}
        self.__unique_Words = []
        self.__result = []
        self.__training_set_size = 0
        self.stop_Words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their',
                           'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

    def fit(self, X, y):
        for [abstract, label] in zip(X, y):
            if label not in self.__data_Class:
                self.__data_Class[label] = 1
            else:
                self.__data_Class[label] += 1
            words = self.__preprecess(abstract.split(" "))
            if label not in self.__each_Class_Words:
                self.__each_Class_Words[label] = words
            else:
                self.__each_Class_Words[label] += words
        self.__training_set_size = len(X)
        self.__calculate_Word_count()
        self.__calculate_Unique_Words()
        self.__calculate_Prob_Of_Class()

    def predict(self, X_test):
        count = 0
        for i in range(0, len(X_test)):
            words = self.__preprecess(X_test[i].split(" "))
            for g in words:
                if g not in self.__unique_Words:
                    count += 1
        result = []
        for sample in X_test:
            result.append(self.__predict_Item(sample, count))
        self.__result = result
        return result

    def __predict_Item(self, abstract, count):
        list_Words = self.__preprecess(abstract.split(" "))
        result = {label: value for label, value in self.__p.items()}
        for word in list_Words:
            for label in result.keys():
                if word in self.__each_Class_Words[label]:
                    result[label] =  math.log((self.__each_Class_Word_Num[label][word]+1) / (self.__data_Class[label] + len(self.__unique_Words) + count))
                else:
                    result[label] += math.log( 1 / (self.__data_Class[label] + len(self.__unique_Words) + count))
        sorted_Result = sorted(result.items(), key = lambda a: a[1], reverse = True)
        # print(sorted_Result)
        return sorted_Result[0][0]

    def __preprecess(self, list_Words):
        result = []
        for word in list_Words:
            # remove stop words
            if word in self.stop_Words:
                continue
            # remove digital words like 1, 2, 3
            if word.isdigit():
                continue
            # remove charactor which is meaningless
            if len(word) == 1:
                continue
            result.append(word)
        # print(result)
        return result

    def results_To_csv(self, fileName):
        df = pd.DataFrame([[id, predict] for [id, predict] in zip(range(1, len(
            self.__result) + 1), self.__result)], columns=["id", "class"]).to_csc(fileName, index=False)

    def __calculate_Prob_Of_Class(self):
        self.__p = {}
        for name in self.__data_Class:
            self.__p[name] = math.log(
                self.__data_Class[name] / self.__training_set_size)
        # print(self.__p)

    def __calculate_Word_count(self):
        for y in self.__data_Class.keys():
            if y not in self.__each_Class_Word_Num:
                self.__each_Class_Word_Num[y] = {}
            for word in self.__each_Class_Words[y]:
                if word not in self.__each_Class_Word_Num[y]:
                    self.__each_Class_Word_Num[y][word] = 1
                else:
                    self.__each_Class_Word_Num[y][word] += 1
        print(self.__each_Class_Word_Num)

    def __calculate_Unique_Words(self):
        words = []
        for y in self.__data_Class.keys():
            words += self.__each_Class_Word_Num[y].keys()
        self.__unique_Words = set(words)
        # print(self.__unique_Words)

    def __str__(self):
        msg = ""
        msg += f"Class in the training set: {self.__data_Class}\n"
        msg += f"Words in each class: {self.__each_Class_Words}\n"
        return msg


training_set = pd.read_csv(os.path.join("data", "trg.csv"))
myCLF = MyBayesCLF()
myCLF.fit(X=training_set["abstract"], y=training_set["class"])

test_Set = pd.read_csv(os.path.join("data", "tst.csv"))
result = myCLF.predict(test_Set["abstract"])
print(result)