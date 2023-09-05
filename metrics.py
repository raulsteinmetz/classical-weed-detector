import pandas as pd


dataset = pd.read_csv('dataset_yolo_v8_ss/runs/segment/train/results.csv')

# print columns names
print(dataset.columns)
print(dataset['metrics/precision(B)'].tail(1))
print(dataset['metrics/recall(B)'].tail(1))