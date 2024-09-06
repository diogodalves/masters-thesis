import os
import shutil
from sklearn.model_selection import train_test_split
import glob

# Step 1: Set Up Directory Structure
base_dir = "data/Latest-Facial-Expression-Recognition/CK+48"
train_dir = os.path.join("data/ck/", 'train')
test_dir = os.path.join("data/ck/", 'test')

# Ensure the directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Step 2: Load and Split Data
# Get all the subdirectories (these are the class labels)
classes = os.listdir(base_dir)

# Iterate over each class directory
for class_name in classes:
    class_dir = os.path.join(base_dir, class_name)

    # Skip if it's the train or test directory we just created
    if class_name in ['train', 'test']:
        continue

    # Get all images in this class
    images = glob.glob(os.path.join(class_dir, '*.png'))  # or '*.jpg', '*.jpeg' depending on image format

    # Split the images into training and testing sets
    train_images, test_images = train_test_split(images, test_size=0.25, random_state=42)

    # Create class subdirectories in train and test directories
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Move the training images to the train directory
    for image in train_images:
        shutil.move(image, os.path.join(train_dir, class_name))

    # Move the testing images to the test directory
    for image in test_images:
        shutil.move(image, os.path.join(test_dir, class_name))

print(f"Data split complete. Training data saved in {train_dir}, and testing data saved in {test_dir}.")
