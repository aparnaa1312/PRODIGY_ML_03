#pil stands for python imaging library, basicalyy used for manipulating with the image unstructured data
#os module interacts with the operating systems
from PIL import Image
import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def pi(image_path,size=(64,64)):
    #convert it to gray_scale(only one channel instead of three colour channels(rgb))
    #l stands for luminance
    img=Image.open(image_path).convert('L')
    #resize to 64*64
    img=img.resize(size)
    img=np.array(img).flatten()
    return img
def train(dir):
    images = []
    labels = []
    for file in os.listdir(dir):
        if file.endswith('.jpg'):
            image_path = os.path.join(dir, file)
            img = pi(image_path)
            images.append(img)
            if 'cat' in file:
                labels.append(1)
            else:
                labels.append(0)
    return images, labels


dir="C:\\Users\\Akshaya Ganesh\\Downloads\\dogs-vs-cats\\train\\train"
images,labels=train(dir)
X=np.array(images)
y=np.array(labels)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train SVM model
clf = svm.SVC()
clf.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")
test="C:\\Users\\Akshaya Ganesh\\Downloads\\dogs-vs-cats\\test1\\test1"
test_images=[]
test_id=[]
for file in os.listdir(test):
        if file.endswith('.jpg'):
            image_path=os.path.join(test,file)
            img=pi(image_path)
            test_images.append(img)
            test_id.append(file.split('.')[0])

X_test = np.array(test_images)

# Predict using the trained SVM model
y_test_pred = clf.predict(X_test)

# Example: Printing out predictions with their image IDs
results = list(zip(test_id, y_test_pred))

# Example of how to print results
for img_id, prediction in results:
    print(f"Image ID: {img_id}, Prediction: {prediction}")
# Load the sample submission file
submission_file_path = "C:\\Users\\Akshaya Ganesh\\Downloads\\dogs-vs-cats\\sampleSubmission.csv"
submission_df = pd.read_csv(submission_file_path)

# Fill the 'label' column with the predictions
submission_df['label'] = y_test_pred

# Save the updated submission file
path="C:\\Users\\Akshaya Ganesh\\Downloads\\submission.csv"
submission_df.to_csv(path)









 
