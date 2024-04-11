import cv2
import os
from imutils import paths
import face_recognition
import pickle

ww = []
yy = []
datapath='dataset'
print("encode start!")
path11 = list(paths.list_images(datapath))
# print(imagePaths)


encodings_path='encodings.pickle'
model1='cnn'   #cnn
for (i, Path2) in enumerate(path11):
    
    print("encode image {}/{}".format(i + 1,
                                                 len(path11)))
    name = Path2.split(os.path.sep)[-2]

    image = cv2.imread(Path2)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model=model1)

    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        # 编码
        ww.append(encoding)
        yy.append(name)

print("encode success!")
data = {"encodings": ww, "names": yy}
f = open(encodings_path, "wb")
f.write(pickle.dumps(data))
f.close()