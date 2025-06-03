import numpy as np
from src.API.constant import POS_INDEX_MAP, AI, HUMAN


# IMPORTANT: 0 is AI, 1 is Human

def convert_pos_str_to_int_array(pos_tags):
    terms = pos_tags.split(",")  # Split the string into a list of strings
    terms = [term.strip() for term in terms]  # Remove leading and trailing whitespaces
    pos_tags_int = []  # Create an empty list to store the integer values

    for term in terms:
        try:
            # Attempt to map the term to its integer value
            pos_tags_int.append(POS_INDEX_MAP[term])
        except KeyError:
            # Handle the case where the term is not found in the map
            print(f"Warning: Term '{term}' not found in POS_INDEX_MAP. Skipping...")
            # Optionally, add a default value like -1 instead of skipping
            # pos_tags_int.append(-1)

    return pos_tags_int


def get_pos_transition_matrix(pos_tags):
    if len(pos_tags) < 2:
        return np.zeros((32, 32))  # Return a zero matrix if input is too short

    num_tags = 32

    # Initialize transition matrix
    pos_transition_matrix = np.zeros((num_tags, num_tags))

    # Count transitions
    for i in range(1, len(pos_tags)):
        pos_transition_matrix[pos_tags[i - 1], pos_tags[i]] += 1

    # Normalize each row to convert counts into probabilities
    row_sums = pos_transition_matrix.sum(axis=1, keepdims=True)
    np.seterr(divide='ignore', invalid='ignore')  # Suppress divide-by-zero warnings
    pos_transition_matrix = np.divide(pos_transition_matrix, row_sums, where=row_sums != 0)
    # Replace NaNs (caused by division by zero) with 0
    pos_transition_matrix = np.nan_to_num(pos_transition_matrix)

    return pos_transition_matrix


class TextData:
    def __init__(self, document_id, embedding_classification, embedding_clustering, content, pos_tags, text_type,
                 model=None):
        """
        Initialize a TextData object.

        :param document_id: A string representing the unique identifier.
        :param embedding_classification: A list of floats representing the large embedding. (3072 floats)
        :param embedding_clustering: A list of floats representing the small embedding. (1536 floats)
        :param content: A string representing the text content.
        :param pos_tags: A list of strings representing POS tags.
        :param text_type: An int indicating the source type ('human' or 'AI model').
        """
        self.id = document_id
        self.embedding_classification = embedding_classification
        self.embedding_clustering = embedding_clustering
        self.small_pca = None
        self.large_pca = None
        self.content = content
        self.pos_tags = convert_pos_str_to_int_array(pos_tags)
        self.pos_transition_matrix = get_pos_transition_matrix(self.pos_tags)
        self.text_type = text_type
        # self.model is null if text_type is 1 (human) and is the model name if text_type is 0 (AI)
        self.model = None if text_type == HUMAN else model
        self.votes = {}  # dict to store votes (0 or 1)

    @classmethod
    def create_from_user_input(cls, content, pos_tags, text_type=HUMAN, classification_embedding=None, model=None):
        """
        Create a TextData object from user input.

        :param content: A string representing the text content.
        :param pos_tags: A list of strings representing POS tags.
        :param text_type: An int indicating the source type ('human' or 'AI model').
        :param model: A string representing the model name (if applicable).
        :return: A TextData object.
        """
        document_id = None
        return cls(document_id, classification_embedding, None, content, pos_tags, text_type, model)

    def add_vote(self, method_name, vote):
        """
        Add a vote to the list of votes.
        :param method_name: A string representing the method name.
        :param vote: An integer (0 or 1) representing a vote.
        """
        if vote not in [0, 1]:
            raise ValueError("Vote must be 0 or 1")
        self.votes[method_name] = vote

    def reset_votes(self):
        """
        Reset the list of votes.
        """
        self.votes = {}

    def classify(self):
        """
        Classify the text based on weighted votes where classifiers have different accuracy levels.
        
        Weights based on accuracy: POS > K-means > Encoder > Agglomerate
        
        :return: An integer (0 or 1) representing the classification (AI or HUMAN).
        """
        if not self.votes:
            raise ValueError("No votes to classify")
        
        # Assign weights based on known accuracy
        weights = {
            'Simple_POS_Transition_Matrix': 4,  # Most accurate
            'KMeans2_embedding_classification': 3,
            'KMeans4_embedding_classification': 3,
            'Auto-Encoder_embedding_classification': 2,
            'Agglomerate258_embedding_classification': 1  # Least accurate
        }
        
        # Calculate weighted sum
        weighted_sum = 0
        total_weight = 0
        
        for method, vote in self.votes.items():
            if method in weights:
                weighted_sum += vote * weights[method]
                total_weight += weights[method]
            else:
                # For any classifier not in our weights dictionary, assign default weight of 1
                weighted_sum += vote * 1
                total_weight += 1
        
        # Normalize the weighted sum (0 to 1 scale)
        normalized_sum = weighted_sum / total_weight
        
        # Decision threshold
        threshold = 0.5
        
        # For debugging
        #print(f"Weighted sum: {weighted_sum}, Total weight: {total_weight}")
        #print(f"Normalized sum: {normalized_sum:.4f}, Threshold: {threshold}")
        
        # In case we're using different constants for AI and HUMAN
        if normalized_sum >= threshold:
            return HUMAN
        else:
            return AI

    def is_wrongly_classified(self):
        """
        Check if the text is wrongly classified based on the votes.

        :return: A boolean indicating whether the text is wrongly classified.
        """
        if not self.votes:
            raise ValueError("No votes to classify")

        # Count occurrences of each vote
        ai_votes = self.votes.count(AI)
        human_votes = self.votes.count(HUMAN)

        # Determine majority
        if ai_votes > human_votes and self.text_type == HUMAN:
            return True
        elif human_votes > ai_votes and self.text_type == AI:
            return True
        else:
            return False

    def __str__(self):
        return (f"TextData(id={self.id}, content={self.content}, pos_tags={self.pos_tags}, "
                f"text_type={self.text_type}, model = {self.model}, votes={self.votes})")

    def __repr__(self):
        return str(self)


def create_text_data_list(documents, text_type):
    """
    Create a list of TextData objects from a list of MongoDB documents.

    :param documents: A list of MongoDB documents.
    :param text_type: An integer indicating the source type ('human' or 'AI model').
    :return: A list of TextData objects.
    """
    text_data_list = []
    for doc in documents:
        text_data = TextData(doc['_id'], doc['text-embedding-005-classification'], doc['text-embedding-005-clustering'],
                             doc['Content'],
                             doc['wordFamilyContent'], text_type, doc['Model'] if text_type == AI else None)
        text_data_list.append(text_data)
    return text_data_list
