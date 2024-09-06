import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import boto3
import io
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from sklearn.preprocessing import LabelEncoder

class CreateDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.s3_client = boto3.client('s3')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        s3_path = self.image_paths[idx]
        image = self.load_image_from_s3(s3_path)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def load_image_from_s3(self, s3_path):
        bucket_name = s3_path.split('/')[2]
        key = '/'.join(s3_path.split('/')[3:])
        
        image_object = self.s3_client.get_object(Bucket=bucket_name, Key=key)
        image_content = image_object['Body'].read()
    
        # Convert the image content to a numpy array for OpenCV
        image_array = np.frombuffer(image_content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError(f"Failed to load image from S3 path: {s3_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        return image
        
def load_images_and_labels(bucket_name, folder_prefix):
    s3_client = boto3.client('s3')
    
    image_paths = []
    labels = []
    continuation_token = None
    
    while True:
        if continuation_token:
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix, ContinuationToken=continuation_token)
        else:
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix)
        
        for item in response.get('Contents', []):
            key = item['Key']
            
            label = key.split('/')[-2]
            
            image_paths.append(f's3://{bucket_name}/{key}')
            labels.append(label)
        
        if response.get('IsTruncated'):
            continuation_token = response.get('NextContinuationToken')
        else:
            break
    
    return image_paths, labels

def encode_labels(y_train,y_test,y_val=None):
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    if y_val is not None:
        y_val = label_encoder.transform(y_val)
        return label_encoder, y_train, y_test, y_val
    return label_encoder, y_train, y_test

def create_data_augmentation(image_size=(224,224)):
    """
    Create data augmentation for training the model.

    Args:
    image_size (tuple): Size to which the images are resized.

    Returns:
    train_transforms (torchvision.transforms.Compose): Transforms for training data.
    val_transforms (torchvision.transforms.Compose): Transforms for validation data.
    """
    return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            ])

def display_augmented_images(image_path, image_size=(224, 224), num_transforms=5):
    """
    Display augmented images using specified transformations.

    Args:
    image_path (str): Path to the image file.
    image_size (tuple): Size to which the images are resized.
    num_transforms (int): Number of transformed images to display.
    """
    original_image = Image.open(image_path)

    transform = create_data_augmentation(image_size)

    def tensor_to_image(tensor):
        return tensor.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, num_transforms + 1, figsize=(15, 5))
    axes[0].imshow(original_image.resize(image_size))
    axes[0].set_title('Original')
    axes[0].axis('off')

    for i in range(num_transforms):
        transformed_tensor = transform(original_image)
        display_image = tensor_to_image(transformed_tensor)
        axes[i + 1].imshow(display_image)
        axes[i + 1].set_title(f'Transformed {i + 1}')
        axes[i + 1].axis('off')

    plt.show()
