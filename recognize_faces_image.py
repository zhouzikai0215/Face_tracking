# import the necessary packages
import face_recognition
import pickle
import cv2

def enhance_image(image):
    # 对比度增强
    alpha = 1.5  # 调整增强的程度
    beta = 30    # 调整亮度
    enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 亮度调整
    brightness_factor = 1.2
    brightened_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

    # 直方图均衡化（增强对比度）
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    equalized_image_rgb = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB)

    # 叠加
    combined_image = cv2.addWeighted(enhanced_image, 1/3, brightened_image, 1/3, 0)
    combined_image = cv2.addWeighted(combined_image, 1, equalized_image_rgb, 1/3, 0)

    return combined_image

encodings_path='encodings.pickle'
your_img='dcc2b6cae4c50dff196e04ea0149d8db.jpeg'
model1='cnn'
print("load encodings start")
data = pickle.loads(open(encodings_path, "rb").read())

image = cv2.imread(your_img)
image = enhance_image(image)
height, width, _ = image.shape
if height < 200 or width < 200:
    image = cv2.resize(image,(600,600))

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("recognize faces")
boxes = face_recognition.face_locations(rgb,model=model1)
encodings = face_recognition.face_encodings(rgb, boxes)
# print(len(encodings))
names = []
# count = 0
for encoding in encodings:
    # count+=1
    # print(count)

    matches = face_recognition.compare_faces(data["encodings"],encoding)
    # print(len(matches))
    # print(matches.count(True))
    name = "Unknown"

    if True in matches:
        # 匹配
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        # 遍历
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1
        values_list = list(counts.values())
        acc = max(values_list)/sum(values_list)
        percentage = acc * 100
        formatted_percentage = "{:.2f}%".format(percentage)
        print(formatted_percentage)
        # print(values_list)
        # print(max(values_list))
        # print(sum(values_list))
        name = max(counts, key=counts.get)

    names.append(name)
    # 遍历人脸
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # 画图
        name2 = 'name:'+name+'   '+'accuracy: '+formatted_percentage
        print(name2)
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name2, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2)
    # show the output image
cv2.imshow("Image", image)
cv2.imwrite("new1.jpg",image)
cv2.waitKey(0)
