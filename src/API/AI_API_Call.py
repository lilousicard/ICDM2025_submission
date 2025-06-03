import json
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from google import genai
from google.genai import types
from openai import OpenAI
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
from src.API.constant import POS_MAP, GEMINI_MODEL_NAME, OPENAI_MODEL_NAME, GEMINI_EMBEDDING_MODEL_NAME, \
    PROJECT_ID, LOCATION

gemini_client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

openai_client = OpenAI()

message_content = f"""
    Write a detailed news article about the given topic. 
    Assume you are a New York Times reporter who just heard of the news. 
    The article must be at least 500 words.
    """


def gemini_embedding_generation(text: str, task="CLASSIFICATION") -> list[float]:
    """Embeds a single text with a pre-trained, foundational model.

    Args:
        text: The input text to be embedded.
        task: The task type for embedding. Check the available tasks in the model's documentation.

    Returns:
        A list containing the embedding vector for the input text.
    """
    # Check if text is empty and raise a more helpful error
    if not text or text.strip() == "":
        raise ValueError("The input text cannot be empty.")

    # Remove new lines and extra spaces
    text = ' '.join(text.split())

    # The dimensionality of the output embedding.
    dimensionality = 256

    model = TextEmbeddingModel.from_pretrained(GEMINI_EMBEDDING_MODEL_NAME)
    input_data = TextEmbeddingInput(text, task)
    kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
    embedding = model.get_embeddings([input_data], **kwargs)[0]

    # Check if the embedding is empty and raise a more helpful error
    if not embedding.values:
        raise ValueError("The embedding values are empty.")

    return embedding.values


def gemini_generation(topic: str, model_name: str = GEMINI_MODEL_NAME):
    """ Generates a news article using the Gemini model.
    Args:
        topic: The topic for the news article.
        model_name: The name of the model to use for generation.
        Returns:
            A tuple containing the title and content of the generated article."""
    si_text1 = message_content

    model = model_name
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=topic)
            ]
        )
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        safety_settings=[types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
        ), types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
        ), types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
        ), types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
        )],
        response_mime_type="application/json",
        response_schema={"type": "OBJECT", "properties": {"Title": {"type": "STRING"}, "Content": {"type": "STRING"}},
                         "required": ["Title", "Content"]},
        system_instruction=[types.Part.from_text(text=si_text1)],
    )

    str = ""

    for chunk in gemini_client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
    ):
        str += chunk.text

    try:
        # Parse the JSON response
        output = json.loads(str)
        title = output.get("Title")
        content = output.get("Content")
        if title is None or content is None:
            raise ValueError("Title or content not found in the response.")
        return title, content
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        raise


def openai_generation(topic: str, model_name: str = OPENAI_MODEL_NAME):
    """ Generates a news article using the OpenAI model.
    Args:
        topic: The topic for the news article.
        model_name: The name of the model to use for generation.
        Returns:
            A tuple containing the title and content of the generated article."""
    response = openai_client.responses.create(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": message_content
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": topic
                    }
                ]
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "document",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The title of the article."
                        },
                        "content": {
                            "type": "string",
                            "description": "The main content of the article."
                        }
                    },
                    "required": [
                        "title",
                        "content"
                    ],
                    "additionalProperties": False
                }
            }
        },
        reasoning={},
        tools=[],
        temperature=1,
        max_output_tokens=2048,
        top_p=1,
        stream=False,
        store=True
    )

    output = json.loads(response.output_text)
    title = output.get("title")
    content = output.get("content")
    if title is None or content is None:
        raise ValueError("Title or content not found in the response.")
    return title, content


def openai_topic_generation(content: str, model_name: str = OPENAI_MODEL_NAME):
    """ Generates a topic using the OpenAI model.
    Args:
        content: The content for which the topic is to be generated.
        model_name: The name of the model to use for generation.
        Returns:
            A string containing the generated topic."""
    response = openai_client.responses.create(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": """You are an archive discovering assistant. 
                        Your task is to extract a concise topic sentence from a news article, 
                        which can be used to prompt a model to generate an article. 
                        Focus on the main subject, key individuals, organizations, or regions, 
                        and the primary event."""
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": content
                    }
                ]
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "document",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The generated topic sentence."
                        }
                    },
                    "required": [
                        "topic"
                    ],
                    "additionalProperties": False
                }
            }
        },
        reasoning={},
        tools=[],
        temperature=1,
        max_output_tokens=2048,
        top_p=1,
        stream=False,
        store=True
    )

    output = json.loads(response.output_text)
    topic = output.get("topic")
    if topic is None:
        raise ValueError("Topic not found in the response.")
    return topic


def words_to_pos_types(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    # Tag each word with its part of speech
    tagged_words = nltk.pos_tag(words)

    # Replace each word with its POS type based on the mapping
    pos_types = [POS_MAP.get(tag, 'Other') for word, tag in tagged_words]
    return ', '.join(pos_types)
