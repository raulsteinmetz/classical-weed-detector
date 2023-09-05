import numpy as n
import pandas as pd
from matplotlib import pyplot as plt


# fixing file
file_path = './dataset_yolo_v8_ss/runs/segment/train/results.csv'

with open(file_path, 'r') as file:
    content = file.read()

content = content.replace(' ', '')

with open(file_path, 'w') as file:
    file.write(content)

training_info = pd.read_csv('./dataset_yolo_v8_ss/runs/segment/train/results.csv')
training_info = training_info[:90]




# SEGMENTATION LOSS TRAINING 

# get dataset columns names
seg_loss = training_info['train/seg_loss']
seg_loss_ma = seg_loss.rolling(10, min_periods=1).mean()

# Plotting the columns
plt.plot(seg_loss, color='blue', label='Segmentation Loss')
plt.plot(seg_loss_ma, color='green', label='Segmentation Loss Moving Average')

# Adding labels and legend
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.savefig('./charts/seg_train.png')

# Display the plot
plt.show()


# SEGMENTATION LOSS VALIDATION

val_seg_loss = training_info['val/seg_loss']
# Clip the values of val_seg_loss_ma to a maximum of 4
val_seg_loss = val_seg_loss.clip(upper=4)
val_seg_loss_ma = val_seg_loss.rolling(10, min_periods=1).mean()


plt.plot(val_seg_loss, label='Validation Segmentation Loss', color='orange')
plt.plot(val_seg_loss_ma, label='Validation Segmentation Loss Moving Average', color='red')

# Adding labels and legend
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.legend()

plt.savefig('./charts/seg_val.png')

# Display the plot
plt.show()


# PRECISION AND RECALL


precision = training_info['metrics/precision(B)']
precision_ma = precision.rolling(10, min_periods=1).mean()
recall = training_info['metrics/recall(B)']
recall_ma = recall.rolling(10, min_periods=1).mean()

# Plotting the columns
plt.plot(precision, label='Precision')
plt.plot(precision_ma, label='Precision Moving Average')

# Adding labels and legend
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()

plt.savefig('./charts/precision.png')

# Display the plot
plt.show()

plt.plot(recall, label='Recall')
plt.plot(recall_ma, label='Recall Moving Average')

# Adding labels and legend
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()

plt.savefig('./charts/recall.png')

plt.show()

# Display the plot
plt.show()



print(training_info.columns)


