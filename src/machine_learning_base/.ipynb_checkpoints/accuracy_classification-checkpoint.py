from torch import nn
from src.API.AI_API_Call import openai_topic_generation, gemini_generation, openai_generation, \
    gemini_embedding_generation, words_to_pos_types
from src.API.text_classifier import TextData
from src.API.constant import AI, HUMAN, OPENAI_MODEL_NAME, GEMINI_MODEL_NAME
from src.API.machine_learning import k_mean_cluster_fit_predict, load_kmeans_model, \
    hierarchical_cluster_fit_predict, load_hierarchical_model,  load_autoencoders, \
    autoencoder_classify_text, evaluate_sequence, load_transition_matrix, Autoencoder


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


class TransitionMatrix:
    def __init__(self, transition_matrix):
        self.transition_matrix = transition_matrix


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



    # Load transition matrix
    human_transition_matrix = load_transition_matrix(human_transition_matrix_file_path)
    ai_transition_matrix = load_transition_matrix(ai_transition_matrix_file_path)

    return kmeans_model, hierarchical_model, h_encoder, AI_encoder, h_optimizer, AI_optimizer, \
              h_criterion, AI_criterion, input_dim, latent_dim, human_transition_matrix, ai_transition_matrix


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
    gemini_response = gemini_generation(topic)
    openai_response = openai_generation(topic)

    # Generate embeddings for the user input
    embedding_response = gemini_embedding_generation(user_input)
    embedding_response_OAI = gemini_embedding_generation(openai_response[1])
    embedding_response_GEMINI = gemini_embedding_generation(gemini_response[1])

    # Generate POS types for the user input
    pos_types = words_to_pos_types(user_input)
    pos_types_OAI = words_to_pos_types(openai_response)
    pos_types_GEMINI = words_to_pos_types(gemini_response)

    # Create text data objects for each response
    text_data = TextData.create_from_user_input(user_input, pos_types, embedding_response)
    text_data_OAI = TextData.create_from_user_input(openai_response, pos_types_OAI, AI, embedding_response_OAI,
                                                    GEMINI_MODEL_NAME)
    text_data_GEMINI = TextData.create_from_user_input(gemini_response, pos_types_GEMINI, AI, embedding_response_GEMINI,
                                                       OPENAI_MODEL_NAME)

    return text_data, text_data_OAI, text_data_GEMINI
