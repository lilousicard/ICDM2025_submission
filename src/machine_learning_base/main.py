import json

import numpy as np
from torch import nn
import sys
from src.API.AI_API_Call import openai_topic_generation, gemini_generation, openai_generation, \
    gemini_embedding_generation, words_to_pos_types
from src.API.text_classifier import TextData
from src.API.constant import AI, HUMAN, OPENAI_MODEL_NAME, GEMINI_MODEL_NAME, EMBEDDING_CLASSIFICATION
from src.API.machine_learning import k_mean_cluster_fit_predict, load_kmeans_model, \
    hierarchical_cluster_fit_predict, load_hierarchical_model, load_autoencoders, \
    autoencoder_classify_text, evaluate_sequence, load_transition_matrix, Autoencoder



def get_file_paths(set_to_use):
    return {
        'kmean': f'src/machine_learning_base/model_file/{set_to_use}/kmean.pkl',
        'hierarchical': f'src/machine_learning_base/model_file/{set_to_use}/hierarchical.pkl',
        'autoencoder': f'src/machine_learning_base/model_file/{set_to_use}/autoencoders.pth',
        'human_matrix': f'src/machine_learning_base/model_file/{set_to_use}/human_matrix.pkl',
        'ai_matrix': f'src/machine_learning_base/model_file/{set_to_use}/AI_matrix.pkl'
    }

class KMeanModel:
    def __init__(self, model, feature_array, cluster_majority):
        self.model = model
        self.feature_array = feature_array
        self.cluster_majority = cluster_majority


class HierarchicalModel:
    def __init__(self, agg_cluster, feature_array, cluster_majority, cluster_centroid):
        self.agg_cluster = agg_cluster
        self.feature_array = feature_array
        self.cluster_majority = cluster_majority
        self.cluster_centroid = cluster_centroid


class AutoencoderModel:
    def __init__(self, encoder, optimizer, criterion, input_dim, latent_dim):
        self.encoder = encoder
        self.optimizer = optimizer
        self.criterion = criterion
        self.input_dim = input_dim
        self.latent_dim = latent_dim


def load_models(
        kmeans_model_file_path: str,
        hierarchical_model_file_path: str,
        autoencoder_file_path: str,
        human_transition_matrix_file_path: str,
        ai_transition_matrix_file_path: str):
    """
    Load the KMeans, Hierarchical, Autoencoder, and transition matrix models.

    """
    # Load KMeans model
    kmeans_model_data = load_kmeans_model(kmeans_model_file_path)
    kmeans_model = KMeanModel(kmeans_model_data['kmeans'], kmeans_model_data['feature_array'],
                              kmeans_model_data['cluster_majorities'])
    # Load Hierarchical model
    hierarchical_model = load_hierarchical_model(hierarchical_model_file_path)
    hierarchical_model = HierarchicalModel(hierarchical_model['agg_cluster'], hierarchical_model['feature_array'],
                                           hierarchical_model['cluster_majorities'],
                                           hierarchical_model['cluster_centroids'])

    # Load Autoencoder model
    # Load the trained autoencoders
    h_encoder, AI_encoder, h_optimizer, AI_optimizer, metadata = load_autoencoders(autoencoder_file_path)

    # Extract input and latent dimensions
    input_dim = metadata["input_dim"]
    latent_dim = metadata["latent_dim"]

    # Define loss functions
    h_criterion = nn.MSELoss()
    AI_criterion = nn.MSELoss()
    # Create Autoencoder model instances
    h_autoencoder = AutoencoderModel(h_encoder, h_optimizer, h_criterion, input_dim, latent_dim)
    AI_autoencoder = AutoencoderModel(AI_encoder, AI_optimizer, AI_criterion, input_dim, latent_dim)

    # Load transition matrix
    human_transition_matrix = load_transition_matrix(human_transition_matrix_file_path)
    ai_transition_matrix = load_transition_matrix(ai_transition_matrix_file_path)

    return kmeans_model, hierarchical_model, h_autoencoder, AI_autoencoder, human_transition_matrix, \
        ai_transition_matrix


def handle_user_input(user_input: str = None):
    """
    Handles user input for the AI API call.
    """
    # Get user input if not provided
    if user_input is None:
        user_input = input("Enter your article you would like to test: ")

    # Generate the topic using the OpenAI model
    topic = openai_topic_generation(user_input)
    print(f"Topic generated: {topic}")
    # Generate the article using the Gemini model
    gemini_response_title, gemini_response_content = gemini_generation(topic)
    openai_response_title, openai_response_content = openai_generation(topic)
    #gemini_response_title, gemini_response_content = "dummy_title", "dummy_content"
    #openai_response_title, openai_response_content = "dummy_title", "dummy_content"

    # Generate embeddings for the user input
    embedding_response = gemini_embedding_generation(user_input)
    embedding_response = np.array(embedding_response).reshape(1, -1)
    embedding_response_OAI = gemini_embedding_generation(openai_response_content)
    embedding_response_OAI = np.array(embedding_response_OAI).reshape(1, -1)
    embedding_response_GEMINI = gemini_embedding_generation(gemini_response_content)
    embedding_response_GEMINI = np.array(embedding_response_GEMINI).reshape(1, -1)

    # Generate POS types for the user input
    pos_types = words_to_pos_types(user_input)
    pos_types_OAI = words_to_pos_types(openai_response_content)
    pos_types_GEMINI = words_to_pos_types(gemini_response_content)

    # Create text data objects for each response
    text_data = TextData.create_from_user_input(user_input, pos_types, classification_embedding=embedding_response)
    text_data_OAI = TextData.create_from_user_input(openai_response_content, pos_types_OAI, AI, embedding_response_OAI,
                                                    OPENAI_MODEL_NAME)
    text_data_GEMINI = TextData.create_from_user_input(gemini_response_content, pos_types_GEMINI, AI, embedding_response_GEMINI,
                                                       GEMINI_MODEL_NAME)

    return text_data, text_data_OAI, text_data_GEMINI


def classify_text(text_data: TextData, k_mean_model: KMeanModel, hierarchical_model: HierarchicalModel,
                  h_autoencoder: AutoencoderModel, AI_autoencoder: AutoencoderModel,
                  human_transition_matrix, ai_transition_matrix):
    """
    Classify the text data using KMeans, Hierarchical clustering, and Autoencoders.
    """
    # KMeans Classification
    k_mean_classification = k_mean_cluster_fit_predict(k_mean_model.model, text_data.embedding_classification,
                                                       k_mean_model.cluster_majority)
    #print(f"KMeans Classification: {k_mean_classification}")
    # Hierarchical Classification
    hierarchical_classification = hierarchical_cluster_fit_predict(hierarchical_model.agg_cluster,
                                                                   text_data.embedding_classification,
                                                                   hierarchical_model.cluster_majority,
                                                                   hierarchical_model.cluster_centroid)
    #print(f"Hierarchical Classification: {hierarchical_classification}")

    # Autoencoder Classification
    autoencoder_classification, hu_pro, ai_pro = autoencoder_classify_text(text_data.embedding_classification, h_autoencoder.encoder,
                                                           AI_autoencoder.encoder, h_autoencoder.criterion,
                                                           AI_autoencoder.criterion, )
    #print(f"Autoencoder Classification: {autoencoder_classification}")

    ai_transition_probability = evaluate_sequence(text_data.pos_tags, ai_transition_matrix)
    human_transition_probability = evaluate_sequence(text_data.pos_tags, human_transition_matrix)
    # if the human_transition_probability is greater than the ai_transition_probability, then the text is classified
    # as human
    if human_transition_probability > ai_transition_probability:
        transition_classification = HUMAN
    else:
        transition_classification = AI

    # Find the classification based on the vote
    classification = text_data.classify_with_vote(k_mean_classification, hierarchical_classification,
                                                  autoencoder_classification, transition_classification)

    return classification, k_mean_classification, hierarchical_classification, autoencoder_classification, \
        transition_classification


def main(user_input_file_path: str = None, set_to_use: str = 'all_text'):
    """
    Main function to load models, handle user input, and classify text.
    """
    # Load models
    file_paths = get_file_paths(set_to_use)
    k_mean_file_path = file_paths['kmean']
    hierarchical_file_path = file_paths['hierarchical']
    autoencoder_file_path = file_paths['autoencoder']
    human_transition_matrix_file_path = file_paths['human_matrix']
    ai_transition_matrix_file_path = file_paths['ai_matrix']
    kmeans_model, hierarchical_model, h_autoencoder, AI_autoencoder, human_transition_matrix, ai_transition_matrix = \
        load_models(k_mean_file_path, hierarchical_file_path, autoencoder_file_path,
                    human_transition_matrix_file_path, ai_transition_matrix_file_path)

    # Read user input for file
    if user_input_file_path is None:
        user_input = None
    else:
        # open the file in read mode
        with open(user_input_file_path, 'r') as file:
            # read the file content
            user_input = file.read()
    text_data, text_data_OAI, text_data_GEMINI = handle_user_input(user_input=user_input)

    # Classify text data (user input)
    classification, k_mean_classification, hierarchical_classification, autoencoder_classification, \
        transition_classification = classify_text(text_data, kmeans_model, hierarchical_model,
                                                  h_autoencoder, AI_autoencoder,
                                                  human_transition_matrix, ai_transition_matrix)
    # Classify text data (OpenAI response)
    classification_OAI, k_mean_classification_OAI, hierarchical_classification_OAI, autoencoder_classification_OAI, \
        transition_classification_OAI = classify_text(text_data_OAI, kmeans_model, hierarchical_model,
                                                        h_autoencoder, AI_autoencoder,
                                                        human_transition_matrix, ai_transition_matrix)
    # Classify text data (Gemini response)
    classification_GEMINI, k_mean_classification_GEMINI, hierarchical_classification_GEMINI, \
        autoencoder_classification_GEMINI, transition_classification_GEMINI = classify_text(text_data_GEMINI,
                                                                                            kmeans_model,
                                                                                            hierarchical_model,
                                                                                            h_autoencoder,
                                                                                            AI_autoencoder,
                                                                                            human_transition_matrix,
                                                                                            ai_transition_matrix)

    # Print the classification results
    print("\n\n")
    print(f"User Input Classification: {'AI' if classification == AI else 'Human'}")
    print(f"KMeans Classification: {'AI' if k_mean_classification == AI else 'Human'}")
    print(f"Hierarchical Classification: {'AI' if hierarchical_classification == AI else 'Human'}")
    print(f"Autoencoder Classification: {'AI' if autoencoder_classification == AI else 'Human'}")
    print(f"Transition Classification: {'AI' if transition_classification == AI else 'Human'}")
    print("\n\n")
    print(f"OpenAI Classification: {'AI' if classification_OAI == AI else 'Human'}")
    print(f"KMeans Classification: {'AI' if k_mean_classification_OAI == AI else 'Human'}")
    print(f"Hierarchical Classification: {'AI' if hierarchical_classification_OAI == AI else 'Human'}")
    print(f"Autoencoder Classification: {'AI' if autoencoder_classification_OAI == AI else 'Human'}")
    print(f"Transition Classification: {'AI' if transition_classification_OAI == AI else 'Human'}")
    print("\n\n")
    print(f"Gemini Classification: {'AI' if classification_GEMINI == AI else 'Human'}")
    print(f"KMeans Classification: {'AI' if k_mean_classification_GEMINI == AI else 'Human'}")
    print(f"Hierarchical Classification: {'AI' if hierarchical_classification_GEMINI == AI else 'Human'}")
    print(f"Autoencoder Classification: {'AI' if autoencoder_classification_GEMINI == AI else 'Human'}")
    print(f"Transition Classification: {'AI' if transition_classification_GEMINI == AI else 'Human'}")
    print("\n\n")


def main_with_json(json_file_path: str, actual_type, output_json_path: str = None, set_to_use: str = 'all_text'):
    wrong_classification = 0
    total_classification = 0
    """
    Alternative main function that processes a JSON file with an array of documents.
    Runs the classification pipeline on each document and stores results back into the JSON.

    Args:
        json_file_path (str): Path to the input JSON file
        output_json_path (str, optional): Path to save the output JSON. If None, will overwrite input file.
    """
    file_paths = get_file_paths(set_to_use)
    k_mean_file_path = file_paths['kmean']
    hierarchical_file_path = file_paths['hierarchical']
    autoencoder_file_path = file_paths['autoencoder']
    human_transition_matrix_file_path = file_paths['human_matrix']
    ai_transition_matrix_file_path = file_paths['ai_matrix']

    # Load models
    kmeans_model, hierarchical_model, h_autoencoder, AI_autoencoder, human_transition_matrix, ai_transition_matrix = \
        load_models(k_mean_file_path, hierarchical_file_path, autoencoder_file_path,
                    human_transition_matrix_file_path, ai_transition_matrix_file_path)

    # Load JSON file
    with open(json_file_path, 'r') as file:
        documents_data = json.load(file)

    # Process each document in the JSON array
    for document in documents_data:
        # Get the document text
        document_text = document.get('Content', '')

        # Process the document through the pipeline
        text_data, text_data_OAI, text_data_GEMINI = handle_user_input(user_input=document_text)

        # Classify text data
        classification, k_mean_classification, hierarchical_classification, autoencoder_classification, \
            transition_classification = classify_text(text_data, kmeans_model, hierarchical_model,
                                                      h_autoencoder, AI_autoencoder,
                                                      human_transition_matrix, ai_transition_matrix)

        if classification != actual_type:
            wrong_classification += 1
        total_classification += 1

        # Store the classification results in the document
        document['research_results'] = {
            'actual_type': actual_type,
            'classification': classification,
            'kmeans_classification': k_mean_classification,
            'hierarchical_classification': hierarchical_classification,
            'autoencoder_classification': autoencoder_classification,
            'transition_classification': transition_classification
        }

    # Determine the output path
    if output_json_path is None:
        output_json_path = json_file_path

    # Save the updated JSON
    with open(output_json_path, 'w') as file:
        json.dump(documents_data, file, indent=4)

    print(f"Processed {len(documents_data)} documents and saved results to {output_json_path}")
    print(f"Wrong Classification: {wrong_classification}")
    print(f"Total Classification: {total_classification}")

    return documents_data


if __name__ == "__main__":
    # Check if the script is run directly
    if len(sys.argv) > 1:
        # If a file path is provided as a command line argument
        user_input_file_path = sys.argv[1]
        main(user_input_file_path=user_input_file_path)
    else:
        # If no file path is provided, run the main function with default parameters
        main()
