import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import shutil
import pywt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Function for wavelet transform
def w2d(img, mode='haar', level=1):
    imArray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray) / 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    return np.uint8(imArray_H)

# Face and eye detection function
def get_cropped_image_if_2_eyes(image_path, face_cascade, eye_cascade):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")  # Log the failed image
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
    return None


# Load cascades
face_cascade = cv2.CascadeClassifier(r"C:\Users\Subasree M B\PycharmProjects\Subasree\Friendscharacterclassifier\model\opencv\haarcascade\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"C:\Users\Subasree M B\PycharmProjects\Subasree\Friendscharacterclassifier\model\opencv\haarcascade\haarcascade_eye.xml")

# Paths to dataset
path_to_data = r"C:\Users\Subasree M B\PycharmProjects\Subasree\Friendscharacterclassifier\model\dataset"
path_to_cr_data = r"C:\Users\Subasree M B\PycharmProjects\Subasree\Friendscharacterclassifier\model\dataset\cropped"

# Create cropped image dataset
img_dirs = [entry.path for entry in os.scandir(path_to_data) if entry.is_dir()]
if os.path.exists(path_to_cr_data):
    shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)

cropped_image_dirs = []
celebrity_file_names_dict = {}
for img_dir in img_dirs:
    celebrity_name = os.path.basename(img_dir)
    celebrity_file_names_dict[celebrity_name] = []
    count = 1
    for entry in os.scandir(img_dir):
        roi_color = get_cropped_image_if_2_eyes(entry.path, face_cascade, eye_cascade)
        if roi_color is not None:
            cropped_folder = os.path.join(path_to_cr_data, celebrity_name)
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_image_dirs.append(cropped_folder)
            cropped_file_name = f"{celebrity_name}{count}.png"
            cropped_file_path = os.path.join(cropped_folder, cropped_file_name)
            cv2.imwrite(cropped_file_path, roi_color)
            celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
            count += 1

# Create class dictionary
class_dict = {celebrity_name: idx for idx, celebrity_name in enumerate(celebrity_file_names_dict.keys())}

# Prepare dataset for training
X, y = [], []
for celebrity_name, training_files in celebrity_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3, 1), scalled_img_har.reshape(32*32, 1)))
        X.append(combined_img)
        y.append(class_dict[celebrity_name])

X = np.array(X).reshape(len(X), 4096).astype(float)
y = np.array(y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model setup
model_params = {
    'svm': SVC(kernel='rbf', C=10, probability=True),
    'random_forest': RandomForestClassifier(),
    'logistic_regression': LogisticRegression(solver='liblinear', multi_class='auto')
}

# Train and evaluate each model
scores = []
best_estimators = {}

for model_name, model in model_params.items():
    pipe = Pipeline([('scaler', StandardScaler()), (model_name, model)])
    pipe.fit(X_train, y_train)
    test_score = pipe.score(X_test, y_test)
    scores.append({'model': model_name, 'test_score': test_score})
    best_estimators[model_name] = pipe

# Convert scores to DataFrame for easy viewing
df = pd.DataFrame(scores, columns=['model', 'test_score'])
print(df)

# Confusion Matrix for the best model (SVM in this case)
best_clf = best_estimators['svm']
y_pred = best_clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

print(classification_report(y_test, y_pred))
print(class_dict)
