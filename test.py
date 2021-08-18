import paddlex as pdx

model = pdx.load_model('模型路径')
image_name = '测试的图片'
result = model.predict(image_name)
pdx.det.visualize(image_name, result, threshold=0.5, save_dir='./output/')
