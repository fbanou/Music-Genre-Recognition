import os
import librosa
import numpy as np
import scipy.io
from sklearn.mixture import GaussianMixture
import joblib
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

DATASET_PATH = "Documents/project/Music-Genres/"  # Path to dataset folder
MFCC_SAVE_PATH = "Documents/project/mfcc_features/"  # Folder for MFCC features
GMM_SAVE_PATH = "Documents/project/gmm_models/"  # Folder to save trained GMM models
NUM_MFCC = 13  # Number of MFCC coefficients
FRAME_SIZE = 0.020  # 20ms window
HOP_LENGTH = 0.005  # 5ms step
MAX_ITER = 100  # Max iterations for EM algorithm
TOL = 1e-6  # Convergence threshold

# Ensure directories exist
os.makedirs(MFCC_SAVE_PATH, exist_ok=True)
os.makedirs(GMM_SAVE_PATH, exist_ok=True)
os.makedirs(os.path.join(MFCC_SAVE_PATH, "train"), exist_ok=True)
os.makedirs(os.path.join(MFCC_SAVE_PATH, "test"), exist_ok=True)

# Function to extract MFCC features from an audio file
def extract_mfcc(file_path):
    # Load the audio file while preserving its original sample rate
    y, sr = librosa.load(file_path, sr=None)

    # Convert frame length to number of samples
    frame_length = int(FRAME_SIZE * sr) 
    # Convert hop size to samples 
    hop_length = int(HOP_LENGTH * sr)  

    # Compute MFCC features
    mfccs = librosa.feature.mfcc(
        y=y,                  # Audio signal
        sr=sr,                # Sample rate
        n_mfcc=NUM_MFCC,      # Number of MFCC coefficients to extract
        n_fft=frame_length,   # Window size for FFT computation
        hop_length=hop_length # Hop size (step between frames)
    )

    # Return the MFCC feature matrix
    return mfccs


# Function to extract MFCC features from each audio file in the dataset and save them in the appropriate folder
def process_dataset(dataset_type):
    # Construct the path to the dataset (either training or testing)
    dataset_path = os.path.join(DATASET_PATH, dataset_type)  
    # Path where extracted MFCCs will be saved
    mfcc_path = os.path.join(MFCC_SAVE_PATH, dataset_type)  

    # Loop through each genre in the dataset
    for genre in os.listdir(dataset_path):
        genre_path = os.path.join(dataset_path, genre)
        genre_mfcc_path = os.path.join(mfcc_path, genre)

        # Skip non-folder files
        if not os.path.isdir(genre_path):
            continue

        # Ensure the directory exists for saving MFCCs
        os.makedirs(genre_mfcc_path, exist_ok=True)

        # Loop through each audio file in the genre folder
        for file in os.listdir(genre_path):
            # Only process '.wav' audio files
            if file.endswith(".wav"):
                file_path = os.path.join(genre_path, file)
                
                # Extract MFCC features from the audio file
                mfcc_features = extract_mfcc(file_path)

                if mfcc_features is not None:
                    # Create a filename with a '.mat' extension for saving MFCCs
                    save_filename = os.path.splitext(file)[0] + ".mat"
                    save_path = os.path.join(genre_mfcc_path, save_filename)

                    # Save extracted MFCC features in MATLAB (.mat) format
                    scipy.io.savemat(save_path, {"mfcc": mfcc_features})
                    
                    # Print confirmation message
                    print(f"Saved: {save_path}")


# Function that loads MFCC feature files for training or testing
def load_mfcc_data(dataset_type):
    # Construct path to MFCC feature folder
    mfcc_path = os.path.join(MFCC_SAVE_PATH, dataset_type)

    # Dictionary to store MFCCs per genre
    genre_mfccs = {}

    # Loop through each genre folder in the MFCC dataset
    for genre in os.listdir(mfcc_path):
        genre_path = os.path.join(mfcc_path, genre)

        # Skip files, process only directories
        if not os.path.isdir(genre_path):
            continue  

        # Loop through each MFCC feature file in the genre folder
        for file in os.listdir(genre_path):
            if file.endswith(".mat"): 
                file_path = os.path.join(genre_path, file)

                # Load the stored MFCC feature matrix
                data = scipy.io.loadmat(file_path)
                
                # Check if MFCC data exists
                if "mfcc" in data: 
                    # Extract the MFCC matrix 
                    mfcc = data["mfcc"] 

                    # Initialize the genre key in dictionary if not already present
                    if genre not in genre_mfccs:
                        genre_mfccs[genre] = []

                    # Transpose MFCCs and store in dictionary
                    genre_mfccs[genre].append(mfcc.T)  

    # Return dictionary of MFCCs grouped by genre
    return genre_mfccs  


# Function to initialize GMM parameters using K-Means clustering
def initialize_gmm(X, n_components):
    # Apply K-Means clustering to group data into n_components clusters
    kmeans = KMeans(n_clusters=n_components, n_init=10, random_state=42)
    # Assigns each sample to a cluster
    labels = kmeans.fit_predict(X) 
    # Use K-Means cluster centers as initial means 
    means = kmeans.cluster_centers_  

    # Compute initial covariance matrices for each cluster
    covariances = np.array([
        np.cov(X[labels == k].T) + np.eye(X.shape[1]) * 1e-6  # Regularization to avoid singularity
        for k in range(n_components)
    ])

    # Compute prior probabilities based on the proportion of data points in each cluster
    cluster_counts = np.bincount(labels, minlength=n_components)  # Count points per cluster
    priors = cluster_counts / X.shape[0]  # Normalize to get probabilities

    # Return the initialized parameters
    return means, covariances, priors


# Function to perform the Expectation step in the EM algorithm for GMM
def e_step(X, means, covariances, priors):
    N = X.shape[0]  # Number of data points
    K = len(priors)  # Number of Gaussian components

    # Initialize the responsibility matrix
    gamma = np.zeros((N, K))

    # Compute responsibilities for each Gaussian component
    for k in range(K):
        # Compute the likelihood of each data point under Gaussian k
        likelihood = multivariate_normal.pdf(X, mean=means[k], cov=covariances[k])
        
        # Multiply by the prior probability of the component
        gamma[:, k] = priors[k] * likelihood  

    # Normalize responsibilities
    gamma /= gamma.sum(axis=1, keepdims=True)  

    # Return the responsibility matrix
    return gamma 


# Function to perform the Maximization step in the EM algorithm for GMM
def m_step(X, gamma):
    # Compute the effective number of points assigned to each cluster
    Nk = gamma.sum(axis=0)  # Sum of responsibilities for each Gaussian component

    # Update means: Compute weighted average of data points assigned to each Gaussian
    means = np.array([
        np.sum(gamma[:, k][:, np.newaxis] * X, axis=0) / Nk[k]  
        for k in range(len(Nk))
    ])

    # Update covariances: Compute weighted covariance matrix for each Gaussian component
    covariances = np.array([
        np.sum(gamma[:, k][:, np.newaxis, np.newaxis] * 
               (X - means[k])[:, :, np.newaxis] @ (X - means[k])[:, np.newaxis, :], axis=0) / Nk[k]
        for k in range(len(Nk))
    ])

    # Update priors: The proportion of points assigned to each Gaussian
    priors = Nk / X.shape[0]

    return means, covariances, priors


# Function to train a Gaussian Mixture Model (GMM) using the Expectation-Maximization (EM) algorithm
def train_gmm(X, n_components):
    # Initialize GMM parameters using K-Means clustering
    means, covariances, priors = initialize_gmm(X, n_components)

    # Initialize previous log-likelihood for convergence check
    prev_log_likelihood = None

    # Iterate through the EM algorithm
    for iteration in range(MAX_ITER):
        
        # E-step: Compute responsibilities based on current GMM parameters
        gamma = e_step(X, means, covariances, priors)  

        # M-step: Update GMM parameters using computed responsibilities
        means, covariances, priors = m_step(X, gamma)  

        # Compute the new log-likelihood to track model convergence
        log_likelihood = np.sum(np.log(np.sum([
            priors[k] * multivariate_normal.pdf(X, mean=means[k], cov=covariances[k])
            for k in range(n_components)
        ], axis=0)))

        # Display log-likelihood progress
        print(f"Iteration {iteration + 1}, Log-Likelihood: {log_likelihood:.6f}")

        # Check for convergence
        if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) < TOL:
            print("Convergence reached.")
            break  # Stop iterations if improvement is below threshold

        prev_log_likelihood = log_likelihood  # Update previous log-likelihood for next iteration

    # Return the optimized GMM parameters
    return means, covariances, priors


# Function that trains a GMM model for each genre and saves the models
def train_gmm_models(genre_mfccs, num_gaussians):
    # Dictionary to store trained GMM models for each genre
    models = {}

    # Iterate over each genre in the dataset
    for genre, mfcc_list in genre_mfccs.items():
        print(f"Training GMM for genre: {genre} with {num_gaussians} Gaussians...")

        # Stack all MFCC feature arrays for this genre
        all_mfccs = np.vstack(mfcc_list)

        # Train the GMM model using the EM algorithm
        means, covariances, priors = train_gmm(all_mfccs, num_gaussians)

        # Save model separately based on K value
        model_filename = os.path.join(GMM_SAVE_PATH, f"gmm_{genre}_K{num_gaussians}.pkl")
        joblib.dump((means, covariances, priors), model_filename)

        models[genre] = (means, covariances, priors)
        print(f"Saved model: {model_filename}")

    # Return trained models
    return models


# Function to load all trained GMM models
def load_gmm_models():
    # Dictionary to store loaded GMM models for each genre
    models = {}

    # Iterate through each saved GMM model file
    for model_file in os.listdir(GMM_SAVE_PATH):
        if model_file.endswith(".pkl"):
            # Extract genre name
            genre = model_file.split("_")[1].split(".")[0]

            # Load the saved GMM parameters
            model_path = os.path.join(GMM_SAVE_PATH, model_file)
            models[genre] = joblib.load(model_path)

            print(f"Loaded model: {model_file} for genre: {genre}")

    # Return the loaded models
    return models


# Function to classify an audio sample using trained GMM models
def classify_audio(mfcc, models):
    # Dictionary to store the log-likelihood for each genre
    log_likelihoods = {}

    # Iterate through each trained GMM model
    for genre, (means, covariances, priors) in models.items():
        
        # Compute log-likelihood for this genre
        likelihoods = np.array([
            priors[k] * multivariate_normal.pdf(mfcc.T, mean=means[k], cov=covariances[k])
            for k in range(len(priors))
        ])

        # Sum over all Gaussian components and take log
        log_likelihoods[genre] = np.sum(np.log(np.sum(likelihoods, axis=0)))

    # Return the genre with the highest log-likelihood
    return max(log_likelihoods, key=log_likelihoods.get)


# Function to evaluate the performance of the trained GMM models on the test dataset
def evaluate_model(models):
    # Initialize counters for accuracy and confusion matrix
    total_samples = 0
    correct_predictions = 0
    confusion_matrix = {}

    # Iterate through each genre in the test dataset
    for genre in os.listdir(os.path.join(DATASET_PATH, "test")):
        genre_path = os.path.join(DATASET_PATH, "test", genre)
        print(f"Processing test samples for genre: {genre}")

        # Skip non-folder files
        if not os.path.isdir(genre_path):
            continue  

        # Initialize confusion matrix entry for this genre
        if genre not in confusion_matrix:
            confusion_matrix[genre] = {}

        # Iterate through each audio file in the genre folder
        for file in os.listdir(genre_path):
            if file.endswith(".wav"):
                total_samples += 1  # Count the total number of test samples

                file_path = os.path.join(genre_path, file)

                # Extract MFCC features from the test file
                mfcc_features = extract_mfcc(file_path)

                if mfcc_features is not None:
                    # Classify the test file
                    predicted_genre = classify_audio(mfcc_features, models)

                    print(f"Testing file: {file}, True genre: {genre}, Predicted genre: {predicted_genre}")

                    # Update confusion matrix
                    if predicted_genre not in confusion_matrix[genre]:
                        confusion_matrix[genre][predicted_genre] = 0

                    confusion_matrix[genre][predicted_genre] += 1

                    # Check if the prediction is correct
                    if predicted_genre == genre:
                        correct_predictions += 1

    # Calculate classification accuracy
    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    print(f"\nModel Classification Accuracy: {accuracy:.2f}%")

    # Print confusion matrix
    print("\nConfusion Matrix:")
    for genre, predictions in confusion_matrix.items():
        print(f"{genre}: {predictions}")

    return accuracy


if __name__ == "__main__":
    print("Extracting MFCCs and saving in train/test folders...")
    process_dataset("train")
    process_dataset("test")

    print("\nLoading training MFCC features...")
    train_mfccs = load_mfcc_data("train")

    # List of different numbers of Gaussians to evaluate
    gaussian_components = [4, 8, 16]
    accuracies = []  # Store accuracy values for plotting

    # Iterate over different Gaussian components
    for num_gaussians in gaussian_components:
        print(f"\nTraining and evaluating GMM with {num_gaussians} Gaussians...\n")

        # Train GMM models with the specified number of Gaussians
        models = train_gmm_models(train_mfccs, num_gaussians)

        # Evaluate the trained models and store accuracy
        accuracy = evaluate_model(models)
        accuracies.append(accuracy)

    print("\nGMM training and classification completed!")

    # Plot accuracy vs. number of Gaussians
    plt.figure(figsize=(8, 5))
    plt.plot(gaussian_components, accuracies, marker='o', linestyle='-')
    plt.xlabel('Number of Gaussians (K)')
    plt.ylabel('Classification Accuracy (%)')
    plt.title('GMM Classification Accuracy vs. Number of Gaussians')
    plt.grid(True)
    plt.xticks(gaussian_components)
    plt.show()

