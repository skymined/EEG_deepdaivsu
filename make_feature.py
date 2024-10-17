import scipy.io as sio
import numpy as np
import os
import torch
import glob

##################################### 코드 읽다 보니까 참고할만한 내용 #########################################
## 1. SEED 데이터에서 제공하는 feature_types는 ["de", "psd", "dasm", "rasm", "asm", "dcau"]. 우리는 de만 사용. ##
## 2. SEED 데이터에서 제공하는 smooth_method_types는 ["movingAve", "LDS1"]. 우리는 movingAve만 사용. 왜지..?    ##
########################################################################################################

dir_path = "ExtractedFeatures/"
feature_types = ["de"]
smooth_method_types = ["movingAve"]
# get labels:
label_path = os.path.join(dir_path, "label.mat")
labels = sio.loadmat(label_path)["label"][0]
files = sorted(glob.glob(dir_path + "*_*"))

sublist = set()
for f in files:
    sublist.add(f.split("/")[-1].split("_")[0])

print("Total number of subjects: {:.0f}".format(len(sublist)))
sublist = sorted(list(sublist))
print(sublist)


##스케일링 코드
def scaling(x):
    mee = np.mean(x, 0)
    x = x - mee
    stdd = np.std(x, 0)
    x = x / (stdd + 1e-7)
    return x


num_of_people = 15
num_of_experiment = 15
for feature_type in feature_types:
    for smooth_method_type in smooth_method_types:
        for i in range(num_of_people):
            print("resolving person {} / {}".format(i + 1, num_of_people))
            stacked_arr = np.zeros([45, 265, 62, 5])
            stacked_label = np.zeros([45, 3])
            stacked_mask = np.zeros([45, 265])
            count = 0
            sub = sublist[i]
            print(f"sub : {sub}")
            sub_files = glob.glob(dir_path + sub + "_*")
            print(f"sub_files : {sub_files}")
            for trial_path in sub_files:
                mov_datai = []
                feature2dict = sio.loadmat(
                    trial_path, verify_compressed_data_integrity=False
                )
                for experiment_index in range(num_of_experiment):
                    k = (
                        feature_type
                        + "_"
                        + smooth_method_type
                        + str(experiment_index + 1)
                    )
                    v = feature2dict[k]
                    data_length = v.shape[1]
                    temp = np.zeros([62, 265, 5])
                    temp_mask = np.zeros([265])
                    temp_mask[: v.shape[1]] = 1
                    temp[:, :data_length, :] = v
                    temp = temp.transpose(1, 0, 2)  ## 1, 0번 자리 바꾸기

                    stacked_arr[count] = temp
                    stacked_label[count, labels[experiment_index] + 1] = 1
                    stacked_mask[count] = temp_mask
                    count = count + 1
            ##스케일링 진행
            stacked_arr = scaling(stacked_arr)
            de_Data = {
                "input": torch.Tensor(stacked_arr),
                "label": torch.Tensor(stacked_label),
                "mask": torch.Tensor(stacked_mask),
            }
            torch.save(de_Data, "SEED_data/test_" + str(i) + "de.pt")

###train dataset 만들기
for i in range(15):
    count = 0
    for sub in range(15):
        if i == sub:
            continue
        elif count == 0:
            total_input = torch.load(
                "SEED_data/test_" + str(sub) + "de.pt", weights_only=True
            )["input"]
            total_label = torch.load(
                "SEED_data/test_" + str(sub) + "de.pt", weights_only=True
            )["label"]
            total_mask = torch.load(
                "SEED_data/test_" + str(sub) + "de.pt", weights_only=True
            )["mask"]
            count = count + 1
        elif count > 0:
            current_input = torch.load(
                "SEED_data/test_" + str(sub) + "de.pt", weights_only=True
            )["input"]
            current_label = torch.load(
                "SEED_data/test_" + str(sub) + "de.pt", weights_only=True
            )["label"]
            current_mask = torch.load(
                "SEED_data/test_" + str(sub) + "de.pt", weights_only=True
            )["mask"]

            total_input = torch.cat([total_input, current_input], dim=0)
            total_label = torch.cat([total_label, current_label], dim=0)
            total_mask = torch.cat([total_mask, current_mask], dim=0)

    Data = {"input": total_input, "label": total_label, "mask": total_mask}
    torch.save(Data, "SEED_data/train_" + str(i) + "de.pt")
