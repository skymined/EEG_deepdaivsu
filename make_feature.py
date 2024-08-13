import scipy.io as sio 
import numpy as np
import os
import torch


dir_path = "ExtractedFeatures"
feature_types = ["de"]
smooth_method_types = ["movingAve"]
# get labels:
label_path = os.path.join(dir_path, "label.mat")
labels = sio.loadmat(label_path)["label"][0]

num_of_people = 15
num_of_experiment = 15
for feature_type in feature_types:
    for smooth_method_type in smooth_method_types:
        folder_name = os.path.join(dir_path, feature_type +"_" + smooth_method_type)
        print("folder name: ", folder_name)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        cumulative_samples = [0]
        for i in range(num_of_people):
            print("resolving person {} / {}".format(i+1, num_of_people))
            stacked_arr = None
            stacked_label = None
            for trial_path in os.listdir(dir_path):
                if trial_path.startswith(str(i+1) + "_"):  # trial record for the person
                    feature2dict= sio.loadmat(os.path.join(dir_path, trial_path))
                    for experiment_index in range(num_of_experiment):
                        k = feature_type + "_" + smooth_method_type + str(experiment_index+1)
                        v = feature2dict[k]
                        temp = np.zeros([1,62,265,5])
                        temp[0,:,:v.shape[1],:] = v   
                        temp_labels = labels[experiment_index]          
                        if stacked_arr is None:
                            stacked_arr = temp.copy()
                            stacked_label = temp_labels.copy()
                        else:
                            stacked_arr = np.vstack((stacked_arr, temp))  # vertically stack arrays
                            stacked_label = np.vstack((stacked_label, temp_labels))
            #de_Data = {"input":torch.Tensor(stacked_arr), "label":torch.Tensor(stacked_label)}
            #torch.save(de_Data,"SEED_data/subject_"+str(i)+"de")
            np.save("SEED_data/subject_"+str(i)+"data",stacked_arr)
            np.save("SEED_data/subject_"+str(i)+"label",stacked_label)