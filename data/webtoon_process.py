from glob import glob
import pandas as pd
import numpy as np

train_test_ratio = 0.95

if __name__ == '__main__':

    files = glob('webtoon/*.jpg')
    files_id = [x.split('_', 1)[0].split('/', 1)[1] for x in files]
    uniq_ids, files_label = np.unique(files_id, return_inverse=True)

    pdfile = pd.DataFrame()
    pdfile['path'] = files[:int(len(files) * train_test_ratio)]
    pdfile['label'] = files_label[:int(len(files) * train_test_ratio)]
    pdfile['id'] = files_id[:int(len(files) * train_test_ratio)]
    pdfile.to_csv('webtoon_process_train.csv', index=False)

    pdfile = pd.DataFrame()
    pdfile['path'] = files[int(len(files) * train_test_ratio):]
    pdfile['label'] = files_label[int(len(files) * train_test_ratio):]
    pdfile['id'] = files_id[int(len(files) * train_test_ratio):]
    pdfile.to_csv('webtoon_process_val.csv', index=False)