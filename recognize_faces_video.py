from imutils.video import VideoStream
import face_recognition
import imutils
import pickle
import time
import cv2

def enhance_image(image):
    # 对比度
    alpha = 1.5  # 增强程度
    beta = 30    # 亮度
    enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 亮度
    brightness_factor = 1.2
    brightened_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

    # 直方图均衡化
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    equalized_image_rgb = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB)

    # 将三种增强效果叠加
    combined_image = cv2.addWeighted(enhanced_image, 1/3, brightened_image, 1/3, 0)
    combined_image = cv2.addWeighted(combined_image, 1, equalized_image_rgb, 1/3, 0)

    return combined_image
encodings_path='encodings.pickle'
output='output7.avi'
display_type=0
model1='cnn'
print("load encodings start")
data = pickle.loads(open(encodings_path, "rb").read())

print("starting video")
vs=cv2.VideoCapture('test3.mp4')
# vs = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# vs.open(0)
writer = None
time.sleep(2.0)

while True:

    ret,frame = vs.read()
    if not ret:
        print("the end")
        break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])

    boxes = face_recognition.face_locations(rgb,
                                            model=model1)
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    for encoding in encodings:

        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)
        name = "Unknown"

        if True in matches:

            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            values_list = list(counts.values())
            acc = max(values_list)/sum(values_list)
            percentage = acc * 100
            formatted_percentage = "{:.2f}%".format(percentage)
            # print(formatted_percentage)
            name = max(counts, key=counts.get)
        names.append(name)
    # 遍历
    for ((top, right, bottom, left), name) in zip(boxes, names):

        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)
        # 画图
        name2 = 'name:'+name+'   '+'accuracy: '+formatted_percentage
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name2, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
        if writer is None and output is not None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output, fourcc, 20,
                                     (frame.shape[1], frame.shape[0]), True)
        if writer is not None:
            writer.write(frame)

        if display_type > 0:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

cv2.destroyAllWindows()

vs.release()

if writer is not None:
    writer.release()
