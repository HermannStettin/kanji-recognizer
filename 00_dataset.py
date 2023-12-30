import struct
from PIL import Image
import numpy as np

def read_record(f):
    s = f.read(576)
    record = struct.unpack(">2H4s504s64s", s)
    img = Image.frombytes("1", (64, 63), record[-2], "raw")
    return record + (img, )

def get_ETL(dataset, num_writers, categories):
    characters = 3036 # characters in dataset
    filename = f"data/ETL9B/{dataset}"
    iter(num_writers)
    iter(categories)
    X, y = [], []
    shape = (64, 64)
    new_img = Image.new("1", shape)
    for writer in num_writers:
        with open(filename, "rb") as f:
            f.seek((writer * characters + 1) * 576)
            for _ in categories:
                record = read_record(f)
                new_img.paste(record[-1], (0, 0))
                out = np.asarray(new_img.getdata()).reshape(shape)
                X.append(out)
                y.append(record[1])
    
    X, y = np.asarray(X, np.int32), np.asarray(y, np.int32)
    return X, y

b_1 = get_ETL("ETL9B_1", range(0, 40), range(0,50))
b_2 = get_ETL("ETL9B_2", range(0, 40), range(0,50))
b_3 = get_ETL("ETL9B_3", range(0, 40), range(0,50))
b_4 = get_ETL("ETL9B_4", range(0, 40), range(0,50))
b_5 = get_ETL("ETL9B_5", range(0, 40), range(0,50))
X = np.concatenate((b_1[0], b_2[0], b_3[0], b_4[0], b_5[0]))
y = np.concatenate((b_1[1], b_2[1], b_3[1], b_4[1], b_5[1]))
np.savez("kanji_imgs.npz", X)
np.savez("kanji_labels.npz", y)