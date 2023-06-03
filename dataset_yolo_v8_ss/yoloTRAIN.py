from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m-seg.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="./data.yaml", epochs=20, batch=1)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
success = model.export(format="torchscript")  # export the model to pt format