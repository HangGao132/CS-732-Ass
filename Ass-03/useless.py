import pandas as pd
import math
def preprecess(list_Words):
        result = []
        stop_Words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
        for word in list_Words:
            # remove stop words
            if word in stop_Words:
                continue
            if len(word) == 1:
                continue
            if word.isdigit():
                continue
            result.append(word)
            
        # print(result)
        return result

training_set = pd.read_csv("./data/trg.csv")
train_class = training_set["class"]
train_abstract = training_set["abstract"]

count_Train_A = 0
count_Train_B = 0
count_Train_E = 0
count_Train_V = 0

length_Of_Train = len(train_class)
A = []
B = []
E = []
V = []

for i in range(0,len(train_class)):
    word = preprecess(train_abstract[i].split(" "))
    # word = train_abstract[i].split(" ")
    if train_class[i] == "A":
        count_Train_A += 1
        A = A + word   
    elif train_class[i] == "B":
        count_Train_B += 1
        B = B + word
    elif train_class[i] == "E":
        count_Train_E += 1
        E = E + word
    else:
        count_Train_V += 1
        V = V + word

dict_a = {}
dict_b = {}
dict_e = {}
dict_v = {}

for w in A:
    if w not in dict_a:
        dict_a[w] = 1
    else:
        dict_a[w] += 1

for w in B:
    if w not in dict_b:
        dict_b[w] = 1
    else:
        dict_b[w] += 1
        
for w in E:
    if w not in dict_e:
        dict_e[w] = 1
    else:
        dict_e[w] += 1
        
for w in V:
    if w not in dict_v:
        dict_v[w] = 1
    else:
        dict_v[w] += 1

unique_words = []
for i in dict_a:
    if i not in unique_words:
        unique_words += [i]
for i in dict_b:
    if i not in unique_words:
        unique_words += [i]
for i in dict_e:
    if i not in unique_words:
        unique_words += [i]
for i in dict_v:
    if i not in unique_words:
        unique_words += [i]
count_a_words = len(A)
count_b_words = len(B)
count_e_words = len(E)
count_v_words = len(V)

def classify(abstracts, dict_a, dict_b, dict_e, dict_v, count_Train_A, count_Train_B, count_Train_E, count_Train_V, unique_words, length_Of_Train, count_a_words, count_b_words, count_e_words, count_v_words):
    uni = len(unique_words)
    na = len(dict_a)
    nb = len(dict_b)
    ne = len(dict_e)
    nv = len(dict_v)
    a_list = []
    num = len(abstracts)
    count = 0
    for i in range(0, num):
        words = preprecess(abstracts[i].split(" ") )
        # words = train_abstract[i].split(" ")
        for g in words:
            if g not in unique_words:
                count += 1
                
    for i in range(0, num):
        words = preprecess(abstracts[i].split(" ") )
        pa = math.log(count_Train_A / length_Of_Train, 2)
        pb = math.log(count_Train_B / length_Of_Train, 2)
        pe = math.log(count_Train_E / length_Of_Train, 2)
        pv = math.log(count_Train_V / length_Of_Train, 2)
        for word in words:
            if word in dict_a:
               
                pa = pa + math.log((dict_a[word]+1) / (count_a_words + uni + count), 2)
            else:
                pa = pa + math.log(1 / (count_a_words + uni + count), 2)
                
            if word in dict_b:
               
                pb = pb + math.log((dict_b[word]+1) / (count_b_words + uni + count), 2)
            else:
                pb = pb + math.log(1 / (count_b_words + uni + count), 2)
                
            if word in dict_e:
                pe = pe + math.log((dict_e[word]+1) / (count_e_words + uni + count), 2)
            else:
                pe = pe + math.log(1 / (count_e_words + uni + count), 2)
                
            if word in dict_v:
                pv = pv + math.log((dict_v[word]+1) / (count_v_words + uni + count), 2)
            else:
                pv = pv + math.log(1 / (count_v_words + uni + count), 2)       
     
        max_num = max(pa, pb, pe, pv)
        if max_num == pa:
            a_list = a_list + ["A"]
        elif max_num == pb:
            a_list = a_list + ["B"]
        elif max_num == pe:
            a_list = a_list + ["E"]
        else:
            a_list = a_list + ["V"]
#     print(a_list)
    return a_list 


test_set = pd.read_csv("./data/tst.csv")
                      
test_set_class_predictions = classify(test_set["abstract"], dict_a, dict_b, dict_e, dict_v, count_Train_A, count_Train_B, count_Train_E, count_Train_V, unique_words, length_Of_Train, count_a_words , count_b_words, count_e_words, count_v_words)
print(test_set_class_predictions)

count = {}
for item in test_set_class_predictions:
    if item in count:
        count[item]+= 1
    else:
        count[item] = 1
print(count)

test_set.insert(1, "class", test_set_class_predictions)
test_set.drop(["abstract"], axis=1).to_csv('out.csv', index=False)  



class KFolds:
    def __init__(self, n_splits, shuffle = True, seed = 4321):
        self.seed = seed
        self.shuffle = shuffle
        self.n_splits = n_splits
        
    def split(self, X):
        num_Of_Samples = X.shape[0]
        indices = np.arange(num_Of_Samples)
        if self.shuffle:
            random_State = np.random.RandomState(self.seed)
            random_State.shuffle(indices)

        for test_mask in self._iter_test_masks(num_Of_Samples, indices):
            train_index = indices[np.logical_not(test_mask)]
            test_index = indices[test_mask]
            yield train_index, test_index
        
    def _iter_test_masks(self, num_Of_Samples, indices):
        fold_sizes = (num_Of_Samples // self.n_splits) * np.ones(self.n_splits, dtype = np.int)
        fold_sizes[:num_Of_Samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            test_mask = np.zeros(num_Of_Samples, dtype = np.bool)
            test_mask[test_indices] = True
            yield test_mask
            current = stop



class KFolds:
    def __init__(self, n_splits, shuffle = True, seed = 4321):
        self.seed = seed
        self.shuffle = shuffle
        self.n_splits = n_splits
    # iterable split function, call it in a for loop and it will iter each train and validation
    def split(self, X):
        num_Of_Samples = X.shape[0]
        indices = np.arange(num_Of_Samples)
        if self.shuffle:
            random_State = np.random.RandomState(self.seed)
            random_State.shuffle(indices)
        for test_mask in self._iter_test_masks(num_Of_Samples, indices):
            train_index = indices[np.logical_not(test_mask)]
            test_index = indices[test_mask]
            train_set = X.filter(train_index, axis = 0)
            test_set = X.filter(test_index, axis = 0)
            yield train_set, test_set
        
    def _iter_test_masks(self, num_Of_Samples, indices):
        fold_sizes = (num_Of_Samples // self.n_splits) * np.ones(self.n_splits, dtype = np.int64)
        fold_sizes[:num_Of_Samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            test_mask = np.zeros(num_Of_Samples, dtype = bool)
            test_mask[test_indices] = True
            yield test_mask
            current = stop