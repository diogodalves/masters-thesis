import os

from typing import List, Tuple, Union

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator

from transformers import ViTFeatureExtractor, ViTModel

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import numpy as np, array
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from tqdm import tqdm

class DataAnalyzer:
    """
    A class for analyzing image data.
    """
    def __init__(self, data_dir: str, train_dir: str = "train"):
        """
        Constructor for DataAnalyzer class.

        Parameters:
            data_dir (str): The directory containing the image data.
        """
        self.data_dir = data_dir
        self.train_dir = os.path.join(self.data_dir, train_dir)

    def load_image_data(self) -> DirectoryIterator:
        """
        Load image data from the specified directory.

        Returns:
            tf.keras.preprocessing.image.DirectoryIterator:
                A tuple containing training iterator.
        """

        train_datagen = ImageDataGenerator(rescale=1./255)

        train_data = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(48, 48),
            batch_size=64,
            class_mode='categorical',
            color_mode="grayscale"
        )

        return train_data

    def display_images_per_label(self, num_images: int = 6) -> None:
        """
        Display images for each label.

        Parameters:
            num_images (int): Number of images to display for each label.
        """

        labels = sorted(os.listdir(self.train_dir))

        fig, axes = plt.subplots(len(labels), num_images, figsize=(10, 10))
        fig.subplots_adjust(hspace=0.5)

        for i, label in enumerate(labels):
            label_dir = os.path.join(self.train_dir, label)
            label_images = os.listdir(label_dir)[:num_images]

            for j, image_name in enumerate(label_images):
                image_path = os.path.join(label_dir, image_name)
                img = image.load_img(image_path, target_size=(48, 48))
                img_array = image.img_to_array(img) / 255.0
                axes[i, j].imshow(img_array.squeeze())
                axes[i, j].set_title(label)
                axes[i, j].axis('off')

        plt.savefig('data-analysis/images.png')
        plt.show()



    def count_files_per_label(self) -> Tuple[List[str], List[int]]:
        """
        Count the number of files per label.

        Returns:
            Tuple[List[str], List[int]]: A tuple containing labels and corresponding file counts.
        """
        labels = sorted(os.listdir(self.train_dir))
        file_counts = []

        for label in labels:
            label_dir = os.path.join(self.train_dir, label)
            num_files = len(os.listdir(label_dir))
            file_counts.append(num_files)

        return labels, file_counts

    def plot_files_per_label(self, labels: List[str], file_counts: List[int]) -> None:
        """
        Plot the number of files per label.

        Parameters:
            labels (List[str]): List of labels.
            file_counts (List[int]): List of file counts corresponding to each label.
        """
        fig = go.Figure(data=[go.Bar(x=labels, y=file_counts)])
        fig.update_layout(title='Number of Files per Label',
                        xaxis_title='Label',
                        yaxis_title='Number of Files')
        fig.show()

    def calculate_image_statistics_by_label(self) -> dict:
        """
        Calculate basic statistics of the images for each label.

        Returns:
            dict: A dictionary containing image statistics (mean, median, min, max, std) for each label.
                  The keys are the label names and the values are dictionaries containing the statistics.
        """
        label_statistics = {}

        for label in os.listdir(self.train_dir):
            label_dir = os.path.join(self.train_dir, label)
            if os.path.isdir(label_dir):
                means = []
                medians = []
                minimums = []
                maximums = []
                stds = []

                for file in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, file)
                    img = plt.imread(img_path)

                    means.append(np.mean(img))
                    medians.append(np.median(img))
                    minimums.append(np.min(img))
                    maximums.append(np.max(img))
                    stds.append(np.std(img))

                label_statistics[label] = {
                    'mean': np.mean(means),
                    'median': np.median(medians),
                    'min': np.min(minimums),
                    'max': np.max(maximums),
                    'std': np.mean(stds)
                }

        return label_statistics

    def plot_statistics_by_label(self, statistics: dict) -> None:
        """
        Plot image statistics by label.

        Parameters:
            statistics (dict): A dictionary containing image statistics by label.
                               The keys are the label names and the values are dictionaries containing the statistics.
        """
        labels = list(statistics.keys())
        statistics_names = list(next(iter(statistics.values())).keys())

        fig = go.Figure()

        for stat_name in statistics_names:
            values = [stats[stat_name] for stats in statistics.values()]
            fig.add_trace(go.Bar(x=labels, y=values, name=stat_name.capitalize()))

        fig.update_layout(
            title='Image Statistics by Label',
            xaxis_title='Label',
            yaxis_title='Value',
            barmode='group'
        )
        fig.show()


    def generate_embeddings(self) -> pd.DataFrame:
        """
        Generate embeddings using a pre-trained Vision Transformer (ViT) model for each image in the dataset.

        Returns:
            pd.DataFrame: DataFrame containing image embeddings along with their respective labels.
        """
        model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

        embeddings = []
        labels = []

        total_images = sum(len(files) for _, _, files in os.walk(self.train_dir))

        pbar = tqdm(total=total_images, desc="Generating embeddings")

        for label in os.listdir(self.train_dir):
            label_dir = os.path.join(self.train_dir, label)
            for file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, file)
                img = Image.open(img_path).convert("RGB")  # Convert to RGB as ViT requires 3-channel images
                inputs = feature_extractor(images=img, return_tensors="pt")

                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

                embeddings.append(embedding)
                labels.append(label)

                pbar.update(1)

        pbar.close()

        embeddings_df = pd.DataFrame({'embeddings': embeddings, 'label': labels})

        return embeddings_df

    def save_embeddings_to_csv(self, embeddings_df: pd.DataFrame, output_file: str) -> None:
        """
        Save image embeddings along with their respective labels to a CSV file.

        Parameters:
            embeddings_df (pd.DataFrame): DataFrame containing image embeddings and labels.
            output_file (str): Path to the output CSV file.
        """
        embeddings_df.to_csv(output_file, index=False)

    def get_original_class_labels(self, data: DirectoryIterator):
        """
        Retrieves the original class labels from a data.

        Parameters
        ----------
        data : DirectoryIterator
            A DirectoryIterator object containing the class indices.

        Returns
        -------
        dict
            A dictionary where keys represent the class indices and values represent
            the corresponding original class labels.
        """
        return {v: k for k, v in data.class_indices.items()}

    def flatten_images(self, data:DirectoryIterator, num_samples_per_class: Union[int, None] = None):
        """
        Flatten images and their labels from a DirectoryIterator object.

        Parameters
        ----------
        data : DirectoryIterator
            A DirectoryIterator object containing the images and their labels.
        num_samples_per_class : Union[int, None], optional
            The number of samples per class to flatten. If None, all samples are flattened.
            Default is None.

        Returns
        -------
        tuple
            A tuple containing two elements:
            - numpy.ndarray: Flattened images.
            - list: Corresponding labels.
        """
        flattened_images = []
        labels = []

        for i in range(len(data)):
            batch = data[i]
            images = batch[0]
            batch_labels = batch[1]

            flattened_batch = images.reshape(images.shape[0], -1)  # Flatten images
            flattened_images.extend(flattened_batch)
            labels.extend(np.argmax(batch_labels, axis=1))

            if num_samples_per_class is not None:
                if len(flattened_images) >= num_samples_per_class * len(np.unique(labels)):
                    break
        return np.array(flattened_images), labels

    def calculate_tsne_perplexity(self,
                                  X: Union[np.ndarray, pd.DataFrame, pd.Series],
                                  min_perplexity: int = 10,
                                  max_perplexity: int = 200,
                                  step: int = 5,
                                  dimensions: int = 2) -> None:
        """
        Calculate the t-SNE divergence for different perplexity values and visualize the results.

        Parameters:
            X (Union[np.ndarray, pd.DataFrame, pd.Series]): The input data.
            dimensions (int): The number of dimensions for the embedding. Default is 2.

        Returns:
            None
        """
        perplexity = np.arange(min_perplexity, max_perplexity, step)
        divergence = []

        for i in perplexity:
            model = TSNE(n_components=dimensions, perplexity=i)
            reduced = model.fit_transform(X)
            divergence.append(model.kl_divergence_)

        fig = go.Figure(data=go.Line(x=perplexity, y=divergence))
        fig.update_layout(title='KL Divergence vs Perplexity',
                        xaxis_title='Perplexity',
                        yaxis_title='KL Divergence')
        fig.show()

    def tsne_visualization(self,
                           data: array,
                           labels: list,
                           class_labels: dict,
                           perplexity: int = 30,
                           dimensions: int = 2) -> None:
        """
        Create a t-SNE visualization of the flattened image data.

        Parameters:
            train_data (DirectoryIterator): The training data iterator.
            num_samples_per_class (int): Number of samples to consider per class.
            dimensions (int): Number of dimensions for t-SNE visualization (2 or 3).
        """
        if dimensions not in [2, 3]:
            raise ValueError("Dimensions must be either 2 or 3.")

        tsne = TSNE(n_components=dimensions,
                    perplexity=perplexity,
                    random_state=42)
        embeddings_tsne = tsne.fit_transform(data)

        fig = go.Figure()

        for class_label_idx in np.unique(labels):
            class_label = class_labels[class_label_idx]
            indices = np.where(np.array(labels) == class_label_idx)[0]
            if dimensions == 2:
                fig.add_trace(go.Scatter(
                    x=embeddings_tsne[indices, 0],
                    y=embeddings_tsne[indices, 1],
                    mode='markers',
                    name=f"{class_label}"
                ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=embeddings_tsne[indices, 0],
                    y=embeddings_tsne[indices, 1],
                    z=embeddings_tsne[indices, 2],
                    mode='markers',
                    name=f"{class_label}"
                ))

        fig.update_layout(
            title=f"t-SNE Visualization of Flattened Image Data ({dimensions}D)",
            legend_title='Class'
        )

        if dimensions == 2:
            fig.update_layout(
                xaxis_title='t-SNE Dimension 1',
                yaxis_title='t-SNE Dimension 2'
            )
        else:
            fig.update_layout(
                scene=dict(
                    xaxis_title='t-SNE Dimension 1',
                    yaxis_title='t-SNE Dimension 2',
                    zaxis_title='t-SNE Dimension 3'
                )
            )

        fig.show()

    def plot_tsne_from_embeddings(self,
                                  X: Union[pd.DataFrame, pd.Series, np.ndarray],
                                  y:Union[pd.DataFrame, pd.Series, np.ndarray],
                                  perplexity: int = 30,
                                  dimensions: int = 2):
        """
        Plot t-SNE visualization from embeddings generated by ViT.

        Args:
            X (Union[pd.DataFrame, pd.Series, np.ndarray]): The input data containing embeddings.
            y (Union[pd.DataFrame, pd.Series, np.ndarray]): The labels corresponding to the embeddings.
            dimensions (int): The number of dimensions for the t-SNE plot. Must be either 2 or 3.

        Returns:
            fig (go.Figure): The Plotly figure object containing the t-SNE visualization.
        """
        if dimensions not in [2, 3]:
            raise ValueError("Dimensions must be either 2 or 3.")

        if dimensions == 2:
            X_tsne = TSNE(n_components=2, perplexity=perplexity).fit_transform(X)
        else:
            X_tsne = TSNE(n_components=3, perplexity=perplexity).fit_transform(X)

        if dimensions == 2:
            tsne_df = pd.DataFrame(X_tsne, columns=['x', 'y'])
        else:
            tsne_df = pd.DataFrame(X_tsne, columns=['x', 'y', 'z'])

        tsne_df['label'] = y

        fig = go.Figure()
        if dimensions == 2:
            for label in tsne_df['label'].unique():
                fig.add_trace(go.Scatter(
                    x=tsne_df[tsne_df['label'] == label]['x'],
                    y=tsne_df[tsne_df['label'] == label]['y'],
                    mode='markers',
                    name=label
                ))
        else:
            for label in tsne_df['label'].unique():
                fig.add_trace(go.Scatter3d(
                    x=tsne_df[tsne_df['label'] == label]['x'],
                    y=tsne_df[tsne_df['label'] == label]['y'],
                    z=tsne_df[tsne_df['label'] == label]['z'],
                    mode='markers',
                    name=label
                ))

        if dimensions == 2:
            fig.update_layout(title='t-SNE Plot (2D)')
        else:
            fig.update_layout(title='t-SNE Plot (3D)')

        return fig

    def plot_pca(self, data: array, labels: list, num_components: int = 2) -> None:
        """
        Plot PCA visualization of the image data using Plotly.

        Parameters:
            X (Union[pd.DataFrame, pd.Series, np.ndarray]): The input data containing embeddings.
            y (Union[pd.DataFrame, pd.Series, np.ndarray]): The labels corresponding to the embeddings.
            dimensions (int): The number of dimensions for the PCA plot. Default is 2.
        """

        pca = PCA(n_components=num_components)
        principal_components = pca.fit_transform(data)

        pca_df = pd.DataFrame(data=principal_components, columns=[f"PC{i+1}" for i in range(num_components)])
        pca_df['label'] = labels

        fig = go.Figure()
        if num_components == 2:
            for label in pca_df['label'].unique():
                fig.add_trace(go.Scatter(
                    x=pca_df[pca_df['label'] == label]['PC1'],
                    y=pca_df[pca_df['label'] == label]['PC2'],
                    mode='markers',
                    name=label
                ))
        else:
            for label in pca_df['label'].unique():
                fig.add_trace(go.Scatter3d(
                    x=pca_df[pca_df['label'] == label]['PC1'],
                    y=pca_df[pca_df['label'] == label]['PC2'],
                    z=pca_df[pca_df['label'] == label]['PC3'],
                    mode='markers',
                    name=label
                ))

        if num_components == 2:
            fig.update_layout(title='PCA Plot (2D)')
        else:
            fig.update_layout(title='PCA Plot (3D)')

        fig.show()

    def plot_pca_embeddings(self,
                            X: Union[pd.DataFrame, pd.Series, np.ndarray],
                            y:Union[pd.DataFrame, pd.Series, np.ndarray],
                            num_components: int = 2) -> None:
            """
            Plot PCA visualization of the image embeddings using Plotly.

            Parameters:
                embeddings_df (pd.DataFrame): DataFrame containing image embeddings and labels.
                num_components (int): Number of principal components for PCA. Default is 2.
            """
            pca = PCA(n_components=num_components)
            principal_components = pca.fit_transform(X)

            pca_df = pd.DataFrame(data=principal_components, columns=[f"PC{i+1}" for i in range(num_components)])
            pca_df['label'] = y

            fig = go.Figure()
            if num_components == 2:
                for label in pca_df['label'].unique():
                    fig.add_trace(go.Scatter(
                        x=pca_df[pca_df['label'] == label]['PC1'],
                        y=pca_df[pca_df['label'] == label]['PC2'],
                        mode='markers',
                        name=label
                    ))
            else:
                for label in pca_df['label'].unique():
                    fig.add_trace(go.Scatter3d(
                        x=pca_df[pca_df['label'] == label]['PC1'],
                        y=pca_df[pca_df['label'] == label]['PC2'],
                        z=pca_df[pca_df['label'] == label]['PC3'],
                        mode='markers',
                        name=label
                    ))

            if num_components == 2:
                fig.update_layout(title='PCA Plot of Embeddings (2D)')
            else:
                fig.update_layout(title='PCA Plot of Embeddings (3D)')

            fig.show()