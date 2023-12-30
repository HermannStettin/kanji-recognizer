import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as Fun

def prepare_data_for_use(features, labels):
    characters = np.load(features)["arr_0"]
    labels = np.load(labels)["arr_0"]

    unique_labels = list(set(labels))
    labels_dict = {unique_labels[i]: i for i in range(len(unique_labels))}
    new_labels = np.array([labels_dict[l] for l in labels], dtype = np.int32)
    
    characters_shuffle, new_labels_shuffle = shuffle(characters, new_labels, random_state = 0)
    X_train, X_test, y_train, y_test = train_test_split(characters_shuffle, new_labels_shuffle, test_size = 0.2, random_state = 42)

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    num_classes = len(unique_labels)
    print(f"Number of classes: {num_classes} ")
    y_train = Fun.one_hot(torch.as_tensor(y_train).long(), num_classes)
    y_test = Fun.one_hot(torch.as_tensor(y_test).long(), num_classes)

    np.savez("kanji_train_imgs.npz", X_train)
    np.savez("kanji_test_imgs.npz", X_test)
    np.savez("kanji_train_labels.npz", y_train)
    np.savez("kanji_test_labels.npz", y_test)

prepare_data_for_use("kanji_features.npz", "kanji_labels.npz")