import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


train_dir = r'C:\Users\sindh\OneDrive\Desktop\Face Mask Detection\Face Mask Dataset\Train'
val_dir = r'C:\Users\sindh\OneDrive\Desktop\Face Mask Detection\Face Mask Dataset\Validation'
for cls in os.listdir(train_dir):
    print(f"{cls}: {len(os.listdir(os.path.join(train_dir, cls)))} images")


datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(train_dir, target_size=(100, 100), batch_size=32, class_mode='binary')
val_data = datagen.flow_from_directory(val_dir, target_size=(100, 100), batch_size=32, class_mode='binary')



model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(train_data, validation_data=val_data, epochs=10)


model.save("mask_detector_model.h5")
print("✅ Model saved as mask_detector_model.h5")
