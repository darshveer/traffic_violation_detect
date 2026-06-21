from roboflow import Roboflow

rf = Roboflow(api_key="qmB9vC2MU6LTHe1JwBpp")
project = rf.workspace("seatbelt-detection-5orx2").project("seatbelt-detection-tlnlh")
version = project.version(8)
dataset = version.download("yolov11")