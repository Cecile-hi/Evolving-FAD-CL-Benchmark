import os
from tqdm import tqdm
source_dir = "fadcl_wav2vec"
feature_dir = "FAD-CL-Benchmark/features"

dataset_list = ["train_fadcl_wav2vec", "test_fadcl_wav2vec"]
# os.mkdir(os.path.join(feature_dir, "train_fadcl_wav2vec"))
for i in range(1,8):
    os.mkdir(os.path.join(feature_dir, "train_fadcl_wav2vec", str(i)))
    os.mkdir(os.path.join(feature_dir, "train_fadcl_wav2vec", str(i), "fake"))
    os.mkdir(os.path.join(feature_dir, "train_fadcl_wav2vec", str(i), "real"))
    with open(source_dir, "label_exp{}.txt".format(i)) as label_f:
        all_info = label_f.readlines()
    for element in tqdm(all_info):
        tmp_label = element.strip().split()[-1]
        subset = element.split()[0].split("/")[-2]
        wav_name = element.split()[0].split("/")[-1]
        if subset == "train":
            if tmp_label == "real":
                os.system("cp {} {}".format(os.path.join(source_dir, "exp{}".format(i), "train", wav_name+".npy"), os.path.join(feature_dir, "train_fadcl_wav2vec", str(i), "real", wav_name.replace(".wav", ".npy"))))
            else:
                os.system("cp {} {}".format(os.path.join(source_dir, "exp{}".format(i), "train", wav_name+".npy"), os.path.join(feature_dir, "train_fadcl_wav2vec", str(i), "fake", wav_name.replace(".wav", ".npy"))))






