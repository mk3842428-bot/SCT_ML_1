import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

BASE_PATH = r'C:\Users\YourUserName\Downloads\archive\animals' 
IMG_SIZE = 64  
LIMIT = 500 

def load_data():
    images = []
    labels = []
    
   
    categories = {'cat': 0, 'dog': 1}
    
    for category, label in categories.items():
        folder_path = os.path.join(BASE_PATH, category)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue
            
        filenames = os.listdir(folder_path)
        print(f"Loading {category} images from: {folder_path} ({len(filenames)} files found)")
        
        count = 0
        for f in filenames:
            if count >= LIMIT:
                break
                
            img_path = os.path.join(folder_path, f)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img.flatten())
                labels.append(label)
                count += 1

    return np.array(images), np.array(labels)


print("Initializing data loading...")
X, y = load_data()

if len(X) > 0:
    print(f"Successfully loaded {len(X)} images. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training SVM (this may take a minute)...")
    model = SVC(kernel='rbf', C=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\nTraining Complete.")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
else:
    print("\nERROR: No images were loaded.")
    print(f"Please double-check that this path is exactly where your folders are: {BASE_PATH}")
