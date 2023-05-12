import cv2, numpy, os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
datasets = 'FaceRecognition/datasets'
# print(os.getcwd())
print('Training')
(images, labels, names, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        print("Subject Path:", subjectpath)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            # print("path:", path)
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

(images, labels) = [numpy.array(lis) for lis in [images, labels]]
# print(images, labels)
(width, height) = (130, 100)
model = cv2.face.LBPHFaceRecognizer_create()
# model = cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)

webcam = cv2.VideoCapture(0)
cnt = 0

# Process input image
while True:
    (_, img) = webcam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))
        # prediction
        prediciton = model.predict(face_resize)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if prediciton[1] < 800:
            cv2.putText(img, '%s - %.0f' % (names[prediciton[0]], prediciton[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
            print(names[prediciton[0]])
            cnt = 0
        else:
            cnt += 1
            cv2.putText(img, 'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255))
            if cnt > 100:
                print("Unknown Person")
                cv2.imwrite("input.jpg", img)
                cnt = 0
    cv2.imshow('Face Recognition', img)
    key = cv2.waitKey(10)
    if key == 27: # 27 for 'esc' key
        break
webcam.release()
cv2.destroyAllWindows()
    