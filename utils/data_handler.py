import random
import numpy as np
import pandas as pd
import copy
import torch
import yaml
import hickle as hkl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


TEST_SIZE = 0.3
TRAIN_VAL_SPLIT = 0.9
NUM_PERMUTATIONS = 3

label_enc_dict = defaultdict(LabelEncoder)

def random_list(num_pair, seed_value):
    random.seed(seed_value)
    a = []
    a.append(random.randrange(6))
    for i in range((num_pair*2)-1):
        x = random.randrange(6)
        while x == a[i]:
            x = random.randrange(6)
        a.append(x)
    return a

def scale_data(X_train, X_test=None, X_val=None):

    flatX_train = X_train.reshape((X_train.shape[0] * X_train.shape[1], X_train.shape[2]))
    s = StandardScaler()
    s.fit(flatX_train)

    print("Standardizing train set")
    flatX_train = s.transform(flatX_train)
    flatX_train = flatX_train.reshape((X_train.shape))

    if X_test is not None:
        print("Standardizing test set")
        flatX_test = X_test.reshape((X_test.shape[0] * X_test.shape[1], X_test.shape[2]))
        flatX_test = s.transform(flatX_test)
        flatX_test = flatX_test.reshape((X_test.shape))

        if X_val is not None:
            print("Standardizing val set")
            flatX_val = X_val.reshape((X_val.shape[0] * X_val.shape[1], X_val.shape[2]))
            flatX_val = s.transform(flatX_val)
            flatX_val = flatX_val.reshape((X_val.shape))

            return flatX_train, flatX_test, flatX_val
        else:
            return flatX_train, flatX_test
    else:
        return flatX_train


class DataHandler:
    def __init__(self, dataname, base_classes, per_batch_classes, seed_value, person):
        self.seed_value = seed_value
        self.seed_randomness()
        self.dataname = dataname
        self.person = person
        self.original_mapping = {}
        self.train_data, self.test_data, self.val_data, self.train_labels, self.test_labels, self.val_labels = self.get_features_and_labels() 
        self.nb_cl = base_classes
        self.num_classes = per_batch_classes
        self.classes_by_groups = []
        self.label_map = {}
        self.train_groups, self.val_groups, self.test_groups = self.initialize() 

    def seed_randomness(self):
        np.random.seed(self.seed_value)

    def get_features_and_labels(self):

        dir_path = 'datasets/' 
        print(dir_path)
        
        if self.dataname == 'uci':
            dataset_name = 'UCI'
        elif self.dataname == 'pamap':
            dataset_name = 'PAMAP'
        elif self.dataname == 'hhar':
            dataset_name = 'HHAR'
        elif self.dataname == 'realworld':
            dataset_name = 'RealWorld_Waist'
        elif self.dataname == 'motion':
            dataset_name = 'MotionSense'
        
        data_path_X = dir_path + 'By_Users/' + dataset_name + '/clientsRawData.hkl'
        data_path_y = dir_path + 'By_Users/' + dataset_name + '/clientsRawLabel.hkl'

        train_data = hkl.load(data_path_X)
        train_labels = hkl.load(data_path_y)    

        print(f"Total users in dataset: {len(train_data)}")

        print('Training data shape:', train_data.shape)
        print('Training label shape:', train_labels.shape)
        
        # Initialize variables to track the maximum variety of activities and corresponding users
        max_variety = 0
        users_with_max_variety = []

        # Iterate through each user's data
        for user_index, user_labels in enumerate(train_labels):
            unique_activities = len(np.unique(user_labels))
            
            # Update the tracking variables if the current user has a higher variety of activities
            if unique_activities > max_variety:
                max_variety = unique_activities
                users_with_max_variety = [user_index]
            elif unique_activities == max_variety:
                users_with_max_variety.append(user_index)

        print('Users', users_with_max_variety, 'have ', max_variety, 'different activities.')

        user_data_sizes = [(user_index, len(train_labels[user_index])) for user_index in users_with_max_variety]
        sorted_users_by_data_size = sorted(user_data_sizes, key=lambda x: x[1], reverse=True)

        print("Users sorted by the total size of their data (in descending order):")
        for user_index, data_size in sorted_users_by_data_size:
            print(f"User {user_index} with data size: {data_size}")

        # Select the user
        selected_user_index = sorted_users_by_data_size[self.person][0] 
        selected_user_data = train_data[selected_user_index]
        selected_user_label = train_labels[selected_user_index]

        print(f"Selected user data shape: {selected_user_data.shape}")
        print(f"Selected user label shape: {selected_user_label.shape}")

        total_size = len(selected_user_label)
        train_size = int(total_size * 0.7)
        test_size = int(total_size * 0.2)

        X_train, temp_data, y_train, temp_labels = train_test_split( selected_user_data, selected_user_label, test_size=total_size-train_size, stratify=selected_user_label, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split( temp_data, temp_labels, test_size=total_size - train_size - test_size, stratify=temp_labels, random_state=42)

        print(f"Training set size: {len(y_train)}")
        print(f"Validation set size: {len(y_val)}")
        print(f"Testing set size: {len(y_test)}")
        unique, counts = np.unique(selected_user_label, return_counts=True)
        activity_counts = dict(zip(unique, counts))

        print("Activity counts for the selected user:", activity_counts)
        
        combined_data_train = zip(X_train, y_train)
        combined_data_test = zip(X_test, y_test)
        combined_data_val = zip(X_val, y_val)

        combined_data_train = sorted(combined_data_train, key=lambda x: x[1])
        combined_data_test = sorted(combined_data_test, key=lambda x: x[1])
        combined_data_val = sorted(combined_data_val, key=lambda x: x[1])

        trainX, trainy, testX, testy  = [], [], [], []
        valX, valy = [], []

        for data, label in combined_data_train:
            trainX.append(data)
            trainy.append(label)
        for data, label in combined_data_test:
            testX.append(data)
            testy.append(label)
        for data, label in combined_data_val:
            valX.append(data)
            valy.append(label)

        X_train = np.array(trainX)
        y_train = np.array(trainy)
        X_test = np.array(testX)
        y_test = np.array(testy)
        X_val = np.array(valX)
        y_val = np.array(valy)
        
        print("X_train.shape: ",X_train.shape)
        print("y_train.shape: ", y_train.shape)
        print("X_test.shape: ", X_test.shape)
        print("y_test.shape: ", y_test.shape)
        print("X_val.shape: ", X_val.shape)
        print("y_val.shape: ", y_val.shape)

        y_train_df = pd.DataFrame(y_train, columns=['AID'])
        y_test_df = pd.DataFrame(y_test, columns=['AID'])
        y_val_df = pd.DataFrame(y_val, columns=['AID'])
        y_train, y_test, y_val = [self.replace_class_labels(df, train=idx == 0) for idx, df in
                           enumerate([y_train_df, y_test_df, y_val_df])]

        X_train = torch.from_numpy(X_train).permute(0,2,1)
        X_test = torch.from_numpy(X_test).permute(0,2,1)
        X_val = torch.from_numpy(X_val).permute(0, 2, 1)

        X_train = X_train.numpy()
        X_test = X_test.numpy()
        X_val = X_val.numpy()

        train_values, test_values, val_values = [df.values.tolist() for df in [y_train_df, y_test_df, y_val_df]]
        y_train, y_test, y_val = [[item[0] for item in each] for each in [train_values, test_values, val_values]]

        return X_train, X_test, X_val, y_train, y_test, y_val

    def get_reversed_original_label_maps(self):
        return dict(map(reversed, self.original_mapping.items()))

    def replace_class_labels(self, df, train=False):
        if train:
            sorted_elements = [i for i in range(len(df.AID.unique()))]
            self.original_mapping = dict(zip(df.AID.unique(), sorted_elements))
            print("self.original_mapping:" ,self.original_mapping)
        df.AID = df.AID.map(self.original_mapping)
        return df

    def get_data_by_groups(self):

        #if train:
        _labels = sorted(set(self.train_labels))
        
        num_classes_per_task = [int(char) for char in self.num_classes]
        
        num_classes_per_task.insert(0, self.nb_cl)
        print('num_classes_per_task: ', num_classes_per_task)

        shuffled_labels = np.random.choice(_labels, len(_labels), replace=False).tolist()
        original_labels = [each for each in range(len(shuffled_labels))]

        self.label_map = dict(zip(shuffled_labels, original_labels))

        print(self.label_map)

        self.classes_by_groups = []
        start_idx = 0

        # Create tasks based on the num_classes_per_task list
        for num_classes in num_classes_per_task:
            if start_idx + num_classes <= len(shuffled_labels):
                self.classes_by_groups.append(shuffled_labels[start_idx:start_idx + num_classes])
                start_idx += num_classes
            else:
                print("Warning: Not enough classes left to create a new task as per the given configuration.")
                break

        self.num_tasks = len(self.classes_by_groups)
        print(f"Classes in each group: {self.classes_by_groups}")
        
        #Training data
        train_class = pd.DataFrame(self.train_labels, columns=['class'])
        train_data_by_classes = {i: [] for i in range(len(set(self.train_labels)))}
        for index, row in enumerate(train_class.iloc):
            train_data_by_classes[int(row['class'])].append(self.train_data[index])

        #Testing data
        test_class = pd.DataFrame(self.test_labels, columns=['class'])
        print('len(set(self.test_labels))', len(set(self.test_labels)))
        test_data_by_classes = {i: [] for i in range(len(set(self.test_labels)))}

        for index, row in enumerate(test_class.iloc):
            test_data_by_classes[int(row['class'])].append(self.test_data[index])

        #Validation data
        val_class = pd.DataFrame(self.val_labels, columns=['class'])
        print('len(set(self.train_labels))', len(set(self.train_labels)))
        val_data_by_classes = {i: [] for i in range(len(set(self.val_labels)))}
        
        for index, row in enumerate(val_class.iloc):
            val_data_by_classes[int(row['class'])].append(self.val_data[index])
            
        train_data_by_class_del = copy.deepcopy(train_data_by_classes)
        test_data_by_class_del = copy.deepcopy(test_data_by_classes)
        val_data_by_class_del = copy.deepcopy(val_data_by_classes)

        client_trainX = [[] for _ in range(self.num_tasks)]
        client_trainy = [[] for _ in range(self.num_tasks)]
        client_testX = [[] for _ in range(self.num_tasks)]
        client_testy = [[] for _ in range(self.num_tasks)]
        client_valX = [[] for _ in range(self.num_tasks)]
        client_valy = [[] for _ in range(self.num_tasks)]
        

        ##########count the number of occurence in each label in train and test dataset
        train_labels_count = {}

        for letter in self.train_labels:
            if letter in train_labels_count:
                train_labels_count[letter] += 1
            else:
                train_labels_count[letter] = 1
        print("train labels_count:", train_labels_count)

        test_labels_count = {}

        for letter in self.test_labels:
            if letter in test_labels_count:
                test_labels_count[letter] += 1
            else:
                test_labels_count[letter] = 1
        print("labels_count:", test_labels_count)
        
        val_labels_count = {}

        for letter in self.val_labels:
            if letter in val_labels_count:
                val_labels_count[letter] += 1
            else:
                val_labels_count[letter] = 1
        print("labels_count:", val_labels_count)

        class_list = set()
        for i in range(self.num_tasks):
            for j in range(len(self.classes_by_groups[i])):
                class_round = self.classes_by_groups[i][j]
                client_trainX[i] += train_data_by_class_del[class_round][:int(train_labels_count[class_round])]
                client_trainy[i] += [class_round] * int(train_labels_count[class_round])
                if class_round not in class_list:
                    client_testX[i] += test_data_by_class_del[class_round][:int(test_labels_count[class_round])]
                    del test_data_by_class_del[class_round][:int(test_labels_count[class_round])]
                    client_testy[i] += [class_round] * int(test_labels_count[class_round])
                    
                    client_valX[i] += val_data_by_class_del[class_round][:int(val_labels_count[class_round])]
                    del val_data_by_class_del[class_round][:int(val_labels_count[class_round])]
                    client_valy[i] += [class_round] * int(val_labels_count[class_round])

                    class_list.add(class_round)
                else:
                    continue
        
        train_grouped_data = [[] for _ in range(self.num_tasks)]
        test_grouped_data = [[] for _ in range(self.num_tasks)]
        val_grouped_data = [[] for _ in range(self.num_tasks)]
        

        
        for i in range(self.num_tasks):
            for data, label in zip(client_trainX[i], client_trainy[i]):
                train_grouped_data[i].append((data, self.label_map[label]))

        for i in range(self.num_tasks):
            for data, label in zip(client_testX[i], client_testy[i]):
                test_grouped_data[i].append((data, self.label_map[label]))

        for i in range(self.num_tasks):
            for data, label in zip(client_valX[i], client_valy[i]):
                val_grouped_data[i].append((data, self.label_map[label]))

        for i in range(self.num_tasks):
            print("print(len(grouped_data[i]))", len(train_grouped_data[i]))
        for i in range(self.num_tasks):
            print("print(len(grouped_data[i]))", len(test_grouped_data[i]))
        for i in range(self.num_tasks):
            print("print(len(grouped_data[i]))", len(val_grouped_data[i]))
        
        print(f"Classes in each group: {self.classes_by_groups}")
        
        return train_grouped_data, val_grouped_data, test_grouped_data

    def initialize(self):
        train_groups, val_groups, test_groups = self.get_data_by_groups()        
        return train_groups, val_groups, test_groups 

    def getNextClasses(self, i):
        return self.train_groups[i], self.val_groups[i], self.test_groups[i]

    def getInputDim(self):
        return self.train_data[0].shape[0]