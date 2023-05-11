from roboflow import Roboflow

rf = Roboflow(api_key='_')
workspace = rf.workspace('_')

model = rf.get_model(model_id='your_model_id_here')

image = rf.image(file='test_image.jpg')
predictions = model.predict(image)
