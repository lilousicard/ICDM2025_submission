import pickle
from collections import Counter
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.API.constant import AI, HUMAN


# Not used in the current implementation
def create_pca(text_data_list, n_components, feature, pca_feature):
    """
    Train a PCA model on a dataset and store the transformed data in each text_data object.

    :param text_data_list: A list of objects representing text data.
    :param n_components: The number of principal components to keep.
    :param feature: The attribute of text_data objects containing features to transform.
    :param pca_feature: The attribute to store the transformed PCA data in each text_data object.
    :return: The trained PCA model.
    """
    # Extract the feature from the text data list
    feature_list = [getattr(text_data, feature) for text_data in text_data_list]

    # Convert the list of features to a numpy array
    feature_array = np.array(feature_list)

    # Train the PCA model
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(feature_array)

    # Store the PCA-transformed data back into the text data
    for i, text_data in enumerate(text_data_list):
        setattr(text_data, pca_feature, pca_data[i])

    # Return the trained PCA model
    return pca


# Not used in the current implementation
def apply_pca(pca_model, text_data_list, feature, pca_feature):
    """
    Apply an already trained PCA model to transform data.

    :param pca_model: A pre-trained PCA model.
    :param text_data_list: A list of objects representing text data.
    :param feature: The attribute of text_data objects containing features to transform.
    :param pca_feature: The attribute to store the transformed PCA data in each text_data object.
    :return: The PCA model (unchanged, for consistency).
    """
    # Extract the feature from the text data list
    feature_list = [getattr(text_data, feature) for text_data in text_data_list]

    # Convert the list of features to a numpy array
    feature_array = np.array(feature_list)

    # Ensure the feature array matches the PCA input dimensions
    if feature_array.shape[1] != pca_model.n_features_in_:
        raise ValueError(
            f"Feature array shape {feature_array.shape[1]} does not match "
            f"PCA input shape {pca_model.n_features_in_}"
        )

    # Transform the data using the trained PCA model
    pca_data = pca_model.transform(feature_array)

    # Store the PCA-transformed data back into the text data
    for i, text_data in enumerate(text_data_list):
        setattr(text_data, pca_feature, pca_data[i])

    return pca_model


def generate_classification_report(y_true, y_pred, target_names=None):
    """
    Generate and print a classification report.

    :param y_true: List or array of true labels.
    :param y_pred: List or array of predicted labels.
    :param target_names: List of class names for the labels (optional).
    :return: Classification report as a string.
    """
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print("Classification Report:")
    print(report)
    return report


def k_mean_cluster(
    text_data_list,
    feature,
    data_type_field,
    n_clusters,
    n_init=10,
    algorithm="lloyd",
    random_state=42,
    file_name="",
):
    """
    Train a KMeans model on a dataset and return the trained model.
    :param random_state: random_state parameter for KMeans.
    :param algorithm: algorithm parameter for KMeans.
    :param n_init: n_init parameter for KMeans.
    :param text_data_list: A list of objects representing text data.
    :param feature: The member variable of text_data objects to use as features.
    :param data_type_field: The member variable of text_data objects to use as the true labels.
    :param n_clusters: The number of clusters to create.
    """
    # Extract the feature from the text data list
    feature_list = [getattr(text_data, feature) for text_data in text_data_list]
    data_type_list = [
        getattr(text_data, data_type_field) for text_data in text_data_list
    ]

    # Convert the list of features to a numpy array
    feature_array = np.array(feature_list)
    print(f"Feature Array Shape: {feature_array.shape}")

    if len(feature_array.shape) > 2:
        feature_array = feature_array.reshape(feature_array.shape[0], -1)
        print(f"Reshaped Feature Array Shape: {feature_array.shape}")

    # Train the KMeans model
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state,
        algorithm=algorithm,
    ).fit(feature_array)
    cluster_labels = kmeans.labels_

    # Count data types in each cluster
    cluster_counts = {}
    for label, data_type in zip(cluster_labels, data_type_list):
        if label not in cluster_counts:
            cluster_counts[label] = Counter()
        cluster_counts[label][data_type] += 1

    # Determine majority type for each cluster
    cluster_majorities = {
        label: counts.most_common(1)[0][0] for label, counts in cluster_counts.items()
    }

    # If all cluster_majorities are the same, print a warning and change the label of the cluster
    # with the most data points to the opposite label
    if len(set(cluster_majorities.values())) == 1:
        print("Warning: All clusters have the same majority label.")
        print(
            "Changing the label of the cluster with the most data points to the opposite label."
        )
        max_cluster = max(cluster_counts, key=lambda x: sum(cluster_counts[x].values()))
        max_cluster_majority = cluster_majorities[max_cluster]
        opposite_label = HUMAN if max_cluster_majority == AI else AI
        cluster_majorities[max_cluster] = opposite_label

    # Assign predicted labels based on majority vote
    y_pred = [cluster_majorities[label] for label in cluster_labels]

    # Generate classification report
    generate_classification_report(data_type_list, y_pred, target_names=["AI", "HUMAN"])
    method_name = "KMeans" + str(n_clusters) + "_" + feature
    # Update `self.votes` for each object
    for text_data, label in zip(text_data_list, cluster_labels):
        # add the majority label (AI or HUMAN) to `self.votes`
        text_data.add_vote(method_name, cluster_majorities[label])

    # Save the model if a filename is provided
    if file_name:
        save_kmeans_model(kmeans, feature_array, cluster_majorities, filename=file_name)

    return kmeans, feature_array, cluster_majorities


def k_mean_cluster_test(
    kmeans_model, text_data_list, feature, data_type_field, cluster_majorities
):
    """
    Test an already trained KMeans model on a dataset, using precomputed cluster majority labels.

    :param kmeans_model: A trained KMeans model.
    :param text_data_list: A list of objects representing text data.
    :param feature: The member variable of text_data objects to use as features.
    :param cluster_majorities: A dictionary mapping cluster labels to majority labels.
    """
    # Extract the feature from the text data list
    feature_list = [getattr(text_data, feature) for text_data in text_data_list]
    true_labels = [getattr(text_data, data_type_field) for text_data in text_data_list]

    # Convert the list of features to a numpy array
    feature_array = np.array(feature_list)

    # Predict clusters using the trained KMeans model
    cluster_labels = kmeans_model.predict(feature_array)

    # Update `vote` for each text_data object based on precomputed cluster_majorities
    predicted_labels = []
    method_name = "KMeans" + str(kmeans_model.n_clusters) + "_" + feature
    for text_data, label in zip(text_data_list, cluster_labels):
        predicted_label = cluster_majorities[label]
        text_data.add_vote(method_name, predicted_label)
        predicted_labels.append(predicted_label)

    # Generate classification report
    generate_classification_report(
        true_labels, predicted_labels, target_names=["AI", "HUMAN"]
    )

    return feature_array, cluster_labels


def k_mean_cluster_fit_predict(
    kmeans_model, embedding, cluster_majorities
):
    """
    Predict the cluster for a single text data object using a trained KMeans model.

    :param kmeans_model: A trained KMeans model.
    :param text_data_obj: An object representing a single text data point.
    :param feature: The member variable of text_data_obj to use as features.
    :param cluster_majorities: A dictionary mapping cluster labels to majority labels.
    """
    # Extract the feature from the text data object

    # Predict the cluster using the trained KMeans model
    cluster_label = kmeans_model.predict(embedding)
    # Update `vote` for the text_data object based on precomputed cluster_majorities
    predicted_label = cluster_majorities[cluster_label[0]]
    return predicted_label


def save_kmeans_model(
    kmeans, feature_array, cluster_majorities, filename="kmeans_model.pkl"
):
    """
    Save the trained KMeans model, feature array, and cluster majorities to a .pkl file.

    :param kmeans: Trained KMeans model.
    :param feature_array: Feature array used in training.
    :param cluster_majorities: Dictionary mapping clusters to majority labels.
    :param filename: Name of the output .pkl file.
    """
    data_to_save = {
        "kmeans": kmeans,
        "feature_array": feature_array,
        "cluster_majorities": cluster_majorities,
    }

    with open(filename, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"Model saved to {filename}")


def load_kmeans_model(filename="kmeans_model.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)


def hierarchical_cluster(
    text_data_list,
    feature,
    data_type_field,
    n_clusters,
    distance_threshold=None,
    file_name="",
):
    """
    Train an Agglomerative (hierarchical) clustering model on a dataset and update votes.

    :param distance_threshold:
    :param text_data_list: A list of text data objects.
    :param feature: The member variable to use as feature.
    :param data_type_field: The member variable holding true labels.
    :param n_clusters: Number of clusters to form.
    :return: A tuple (agg_cluster_model, feature_array, cluster_majorities, cluster_centroids)
    """
    # Extract features and true labels
    feature_list = [getattr(text_data, feature) for text_data in text_data_list]
    data_type_list = [
        getattr(text_data, data_type_field) for text_data in text_data_list
    ]

    # Convert to numpy array and reshape if needed
    feature_array = np.array(feature_list)
    print(f"Feature Array Shape: {feature_array.shape}")
    if len(feature_array.shape) > 2:
        feature_array = feature_array.reshape(feature_array.shape[0], -1)
        print(f"Reshaped Feature Array Shape: {feature_array.shape}")

    # Perform Agglomerative Clustering
    agg_cluster = AgglomerativeClustering(
        n_clusters=n_clusters if distance_threshold is None else None,
        linkage="ward",
        compute_full_tree="auto",
        distance_threshold=distance_threshold,
    )
    cluster_labels = agg_cluster.fit_predict(feature_array)

    # Count data types in each cluster
    cluster_counts = {}
    for label, data_type in zip(cluster_labels, data_type_list):
        if label not in cluster_counts:
            cluster_counts[label] = Counter()
        cluster_counts[label][data_type] += 1

    # Determine majority type for each cluster
    cluster_majorities = {
        label: counts.most_common(1)[0][0] for label, counts in cluster_counts.items()
    }

    # If all clusters have the same majority label, change the label of the largest cluster
    if len(set(cluster_majorities.values())) == 1:
        print("Warning: All clusters have the same majority label.")
        print(
            "Changing the label of the cluster with the most data points to the opposite label."
        )
        max_cluster = max(cluster_counts, key=lambda x: sum(cluster_counts[x].values()))
        max_cluster_majority = cluster_majorities[max_cluster]
        opposite_label = HUMAN if max_cluster_majority == AI else AI
        cluster_majorities[max_cluster] = opposite_label

    # Update votes for each text_data object
    method_name = "Agglomerative" + str(n_clusters) + "_" + feature
    for text_data, label in zip(text_data_list, cluster_labels):
        text_data.add_vote(method_name, cluster_majorities[label])

    # Generate classification report based on majority votes
    predicted_labels = [cluster_majorities[label] for label in cluster_labels]
    generate_classification_report(
        data_type_list, predicted_labels, target_names=["AI", "HUMAN"]
    )

    # Compute centroids for each cluster to enable prediction on new data
    cluster_centroids = {}
    for label in np.unique(cluster_labels):
        cluster_centroids[label] = feature_array[cluster_labels == label].mean(axis=0)

    # Save the model if a filename is provided
    if file_name:
        save_hierarchical_model(
            agg_cluster,
            feature_array,
            cluster_majorities,
            cluster_centroids,
            filename=file_name,
        )

    return agg_cluster, feature_array, cluster_majorities, cluster_centroids


def hierarchical_cluster_test(
    agg_cluster_model,
    text_data_list,
    feature,
    data_type_field,
    cluster_majorities,
    cluster_centroids,
):
    """
    Test hierarchical clustering on new data by assigning each sample to the nearest centroid.

    :param agg_cluster_model: The (already trained) AgglomerateClustering model (not used for prediction).
    :param text_data_list: A list of text data objects.
    :param feature: The member variable to use as feature.
    :param data_type_field: The member variable holding true labels.
    :param cluster_majorities: Dictionary mapping cluster label to majority label.
    :param cluster_centroids: Dictionary mapping cluster label to centroid (numpy array).
    :return: A tuple (feature_array, predicted_cluster_labels)
    """
    # Extract features and true labels
    feature_list = [getattr(text_data, feature) for text_data in text_data_list]
    true_labels = [getattr(text_data, data_type_field) for text_data in text_data_list]
    feature_array = np.array(feature_list)
    if len(feature_array.shape) > 2:
        feature_array = feature_array.reshape(feature_array.shape[0], -1)

    predicted_cluster_labels = []
    predicted_labels = []
    method_name = "Agglomerate" + str(len(cluster_majorities)) + "_" + feature

    # Assign each test sample to the nearest centroid
    for i, vec in enumerate(feature_array):
        # Compute Euclidean distances to each centroid
        distances = {
            label: norm(vec - centroid) for label, centroid in cluster_centroids.items()
        }
        assigned_cluster = min(distances, key=distances.get)
        predicted_cluster_labels.append(assigned_cluster)

        predicted_label = cluster_majorities[assigned_cluster]
        predicted_labels.append(predicted_label)
        text_data_list[i].add_vote(method_name, predicted_label)

    generate_classification_report(
        true_labels, predicted_labels, target_names=["AI", "HUMAN"]
    )

    return feature_array, predicted_cluster_labels


def hierarchical_cluster_fit_predict(
    agg_cluster_model, embedding, cluster_majorities, cluster_centroids
):
    """
    Predict the cluster for a single text data object using a trained Agglomerative Clustering model.

    :param agg_cluster_model: A trained AgglomerativeClustering model.
    :param text_data_obj: An object representing a single text data point.
    :param feature: The member variable of text_data_obj to use as features.
    :param cluster_majorities: A dictionary mapping cluster labels to majority labels.
    :param cluster_centroids: A dictionary mapping cluster labels to centroids.
    """

    # Compute Euclidean distances to each centroid
    distances = {
        label: norm(embedding - centroid)
        for label, centroid in cluster_centroids.items()
    }
    assigned_cluster = min(distances, key=distances.get)

    # Update `vote` for the text_data object based on precomputed cluster_majorities
    predicted_label = cluster_majorities[assigned_cluster]

    return predicted_label


def save_hierarchical_model(
    agg_cluster,
    feature_array,
    cluster_majorities,
    cluster_centroids,
    filename="hierarchical_model.pkl",
):
    """
    Save the trained Agglomerative Clustering model, feature array, cluster majorities, and centroids to a .pkl file.

    :param agg_cluster: Trained Agglomerative Clustering model.
    :param feature_array: Feature array used in training.
    :param cluster_majorities: Dictionary mapping clusters to majority labels.
    :param cluster_centroids: Dictionary mapping clusters to centroids.
    :param filename: Name of the output .pkl file.
    """
    data_to_save = {
        "agg_cluster": agg_cluster,
        "feature_array": feature_array,
        "cluster_majorities": cluster_majorities,
        "cluster_centroids": cluster_centroids,
    }

    with open(filename, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"Model saved to {filename}")


def load_hierarchical_model(filename="hierarchical_model.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(), nn.Linear(1024, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.ReLU(), nn.Linear(1024, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


def create_dataloader(text_data_list, feature_field, batch_size):
    """
    Create a DataLoader object for a list of text data objects.
    :param text_data_list: A list of objects representing text data.
    :param feature_field: The attribute of text_data objects containing features.
    :param batch_size: The batch size for the DataLoader.
    :return: A DataLoader object.
    """
    tensor = torch.tensor(
        [getattr(data, feature_field) for data in text_data_list], dtype=torch.float32
    )
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def create_autoencoder(input_dim, latent_dim, learning_rate=0.001):
    """
    Create an autoencoder model, loss function, and optimizer.
    :param input_dim: The dimension of the input data.
    :param latent_dim: The dimension of the latent space.
    :param learning_rate: The learning rate for the optimizer.

    :return: A tuple (autoencoder, criterion, optimizer).
    """
    autoencoder = Autoencoder(input_dim, latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    return autoencoder, criterion, optimizer


def train_autoencoder(autoencoder, criterion, optimizer, dataloader, num_epochs):
    """
    Train an autoencoder model on a dataset.
    :param autoencoder: The autoencoder model to train.
    :param criterion: The loss function to use.
    :param optimizer: The optimizer to use.
    :param dataloader: The DataLoader object containing the training data.
    :param num_epochs: The number of epochs to train for.
    """
    for epoch in range(num_epochs):
        total_loss = 0
        for data in dataloader:
            inputs = data[0]
            # Forward pass
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")


def autoencoder_classify_text(
    embedding, h_encoder, AI_encoder, h_criterion, AI_criterion
):
    """
    Classify a single text embedding using two trained autoencoders.

    Parameters:
    - embedding (Array of Floats): The input text embedding to classify.
    - h_encoder (nn.Module): Trained autoencoder for human text.
    - AI_encoder (nn.Module): Trained autoencoder for AI-generated text.
    - h_criterion (nn.Module): Loss function for human autoencoder.
    - AI_criterion (nn.Module): Loss function for AI autoencoder.

    Returns:
    - classification (str): The predicted label (HUMAN or AI).
    - error_human (float): Reconstruction error for the human autoencoder.
    - error_ai (float): Reconstruction error for the AI autoencoder.
    """
    embedding = torch.tensor(embedding, dtype=torch.float32)
    with torch.no_grad():
        # Get reconstruction errors
        reconstructed_human = h_encoder(embedding)
        error_human = h_criterion(reconstructed_human, embedding).item()

        reconstructed_ai = AI_encoder(embedding)
        error_ai = AI_criterion(reconstructed_ai, embedding).item()

        # Classify based on reconstruction error
        classification = HUMAN if error_human < error_ai else AI

    return classification, error_human, error_ai


def save_autoencoders(
    h_encoder,
    AI_encoder,
    h_optimizer,
    AI_optimizer,
    input_dim,
    latent_dim,
    filename="autoencoders.pth",
):
    """
    Save the trained autoencoders, optimizers, and metadata to a file.

    :param h_encoder: Trained human autoencoder.
    :param AI_encoder: Trained AI autoencoder.
    :param h_optimizer: Optimizer for human autoencoder.
    :param AI_optimizer: Optimizer for AI autoencoder.
    :param input_dim: Input feature size.
    :param latent_dim: Latent space size.
    :param filename: File to save the models.
    """
    save_data = {
        "h_encoder_state": h_encoder.state_dict(),
        "AI_encoder_state": AI_encoder.state_dict(),
        "h_optimizer_state": h_optimizer.state_dict(),
        "AI_optimizer_state": AI_optimizer.state_dict(),
        "input_dim": input_dim,
        "latent_dim": latent_dim,
    }

    torch.save(save_data, filename)
    print(f"Autoencoders and metadata saved to {filename}")


def load_autoencoders(filename="autoencoders.pth"):
    """
    Load the autoencoders, optimizers, and metadata from a file.

    :param filename: File to load the models from.
    :return: Tuple (h_encoder, AI_encoder, h_optimizer, AI_optimizer, metadata)
    """
    metadata = torch.load(filename)

    # Recreate models and optimizers
    h_encoder, _, h_optimizer = create_autoencoder(
        metadata["input_dim"], metadata["latent_dim"]
    )
    AI_encoder, _, AI_optimizer = create_autoencoder(
        metadata["input_dim"], metadata["latent_dim"]
    )

    # Load saved states
    h_encoder.load_state_dict(metadata["h_encoder_state"])
    AI_encoder.load_state_dict(metadata["AI_encoder_state"])
    h_optimizer.load_state_dict(metadata["h_optimizer_state"])
    AI_optimizer.load_state_dict(metadata["AI_optimizer_state"])

    print("Autoencoders successfully loaded")
    return h_encoder, AI_encoder, h_optimizer, AI_optimizer, metadata


def compute_transition_matrix(sequences, n_states=32, smoothing=1e-6):
    """
    Compute the transition probability matrix from training sequences.

    Parameters:
    - sequences (list of lists): Each inner list contains POS tags as integers.
    - n_states (int): Total number of states.

    Returns:
    - transition_matrix (numpy.ndarray): A matrix of shape (n_states, n_states).
    """
    # Initialize a matrix to count transitions
    transition_counts = np.zeros((n_states, n_states))

    # Count transitions
    for seq in sequences:
        for i in range(len(seq) - 1):
            transition_counts[seq[i], seq[i + 1]] += 1

    # Add smoothing to avoid zero probabilities
    transition_counts += smoothing

    # Normalize to get probabilities
    transition_matrix = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    transition_matrix = np.nan_to_num(transition_matrix)  # Handle division by zero

    return transition_matrix


def evaluate_sequence(sequence, transition_matrix):
    """
    Compute the log-likelihood of a sequence given a transition matrix.

    Parameters:
    - sequence (list of ints): A sequence of states.
    - transition_matrix (numpy.ndarray): A matrix of shape (n_states, n_states).

    Returns:
    - log_likelihood (float): The log-likelihood of the sequence.
    """

    log_likelihood = 0.0

    for i in range(len(sequence) - 1):
        prob = transition_matrix[sequence[i], sequence[i + 1]]
        if prob > 0:
            log_likelihood += np.log(prob)
        else:
            log_likelihood += np.log(1e-10)  # Small value for zero probabilities

    return log_likelihood / (len(sequence) - 1)




def save_transition_matrix(transition_matrix, filename="transition_matrix.pkl"):
    """
    Save the transition matrix to a file using pickle.

    Parameters:
    - transition_matrix (numpy.ndarray): The computed transition probability matrix.
    - filename (str): The name of the file to save.
    """
    with open(filename, "wb") as f:
        pickle.dump(transition_matrix, f)

    print(f"Transition matrix saved to {filename}")


def load_transition_matrix(filename="transition_matrix.pkl"):
    """
    Load the transition matrix from a file using pickle.

    Parameters:
    - filename (str): The name of the file to load.

    Returns:
    - transition_matrix (numpy.ndarray): The loaded transition probability matrix.
    """
    with open(filename, "rb") as f:
        transition_matrix = pickle.load(f)

    print(f"Transition matrix loaded from {filename}")
    return transition_matrix


def plot_actual_label(text_data_list, feature, title="Actual Classification"):
    """
    Plot the actual classification of the data.

    :param text_data_list: A list of objects representing text data.
    :param feature: The attribute of text_data objects containing features to plot.
    :param title: Title for the plot.
    """
    # Get the actual classification (AI or HUMAN)
    labels = [
        text_data.text_type for text_data in text_data_list
    ]  # Assuming text_type stores AI or HUMAN constants

    # Extract the feature from the text data list
    feature_list = [getattr(text_data, feature) for text_data in text_data_list]

    # Convert the list of features to a numpy array
    feature_array = np.array(feature_list)

    # Ensure the feature array has at least two dimensions
    if feature_array.shape[1] < 2:
        raise ValueError(
            "Feature array must have at least two dimensions for plotting."
        )

    # Map AI and HUMAN to colors
    color_map = {AI: "blue", HUMAN: "orange"}
    point_colors = [color_map[label] for label in labels]

    # Scatter plot for the features
    plt.figure(figsize=(10, 6))
    plt.scatter(
        feature_array[:, 0],
        feature_array[:, 1],
        c=point_colors,
        alpha=0.5,
        label="Data Points",
    )

    # Create a custom legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=10,
            label="AI",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="orange",
            markersize=10,
            label="HUMAN",
        ),
    ]
    plt.legend(handles=legend_elements, title="Classification")

    # Adding labels and title
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def plot_actual_label_3d(text_data_list, feature, title="Actual Classification (3D)"):
    """
    Plot the actual classification of the data in 3D.

    :param text_data_list: A list of objects representing text data.
    :param feature: The attribute of text data objects containing features to plot.
    :param title: Title for the plot.
    """
    # Get the actual classification (AI or HUMAN)
    labels = [
        text_data.text_type for text_data in text_data_list
    ]  # Assuming text_type stores AI or HUMAN constants

    # Extract the feature from the text data list
    feature_list = [getattr(text_data, feature) for text_data in text_data_list]

    # Convert the list of features to a numpy array
    feature_array = np.array(feature_list)

    # Ensure the feature array has at least three dimensions
    if feature_array.shape[1] < 3:
        raise ValueError(
            "Feature array must have at least three dimensions for 3D plotting."
        )

    # Map AI and HUMAN to colors
    color_map = {AI: "blue", HUMAN: "orange"}  # Adjust according to your constants
    point_colors = [color_map[label] for label in labels]

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot
    ax.scatter(
        feature_array[:, 0],  # X-axis
        feature_array[:, 1],  # Y-axis
        feature_array[:, 2],  # Z-axis
        c=point_colors,
        alpha=0.6,
        label="Data Points",
    )

    # Create a custom legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=10,
            label="AI",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="orange",
            markersize=10,
            label="HUMAN",
        ),
    ]
    ax.legend(handles=legend_elements, title="Classification")

    # Adding labels and title
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")

    # Show plot
    plt.show()


def plot_latest_result(
    data_list, method_name, feature_field, title="Latest Classification Result"
):
    """
    Visualize data points based on a specified feature and their latest vote.

    :param data_list: List of TextData objects containing features and votes.
    :param method_name: Name of the method used for the latest vote.
    :param feature_field: String name of the feature attribute to plot.
    :param title: Title for the plot.
    """
    # Extract features and latest votes
    features = [getattr(data, feature_field) for data in data_list]
    # fetch from the votes dictionary the method_name vote
    latest_votes = [
        data.votes[method_name] if method_name in data.votes else None
        for data in data_list
    ]

    # Filter out data points without votes
    filtered_data = [
        (feat, vote) for feat, vote in zip(features, latest_votes) if vote is not None
    ]
    if not filtered_data:
        raise ValueError("No data points with votes available for plotting.")

    # Separate features and votes after filtering
    features, latest_votes = zip(*filtered_data)

    # Convert features to a NumPy array for easier indexing
    feature_array = np.array(features)

    # Ensure the feature array has at least two dimensions
    if feature_array.shape[1] < 2:
        raise ValueError(
            "Feature array must have at least two dimensions for plotting."
        )

    # Map votes to colors
    color_map = {AI: "blue", HUMAN: "orange"}
    point_colors = [color_map[vote] for vote in latest_votes]

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        feature_array[:, 0], feature_array[:, 1], c=point_colors, alpha=0.5
    )

    # Create custom legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=10,
            label="AI",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="orange",
            markersize=10,
            label="HUMAN",
        ),
    ]
    plt.legend(handles=legend_elements, title="Classification")

    # Add labels and title
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
