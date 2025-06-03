# Running the Thesis Project

This document outlines the steps required to set up and run the Python project for this thesis.

## Prerequisites

* **Python 3.12** installed on your system.
* **pip** (Python package installer) should be installed along with Python.

## Installation

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone URL_TO_REPOSITORY
    cd repository_name
    ```

2.  **Create a virtual environment:**
    It's highly recommended to create a virtual environment to isolate the project dependencies.
    ```bash
    python3 -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **On macOS and Linux:**
        ```bash
        source venv/bin/activate
        ```
    * **On Windows:**
        ```bash
        venv\Scripts\activate
        ```

4.  **Install required dependencies:**
    With the virtual environment activated, install the necessary Python libraries from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Set the OpenAI API Key:**
    This project utilizes the OpenAI API. You need to export your API key as an environment variable:
    ```bash
    export OPENAI_API_KEY="your_openai_api_key_here"
    ```
    *(Replace `"your_openai_api_key_here"` with your actual OpenAI API key.)*

2.  **Configure Google Cloud Project ID:**
    Navigate to the following file:
    ```
    src/API/constant.py
    ```
    Open this file and locate the lines related to the Google Cloud project ID. Modify the placeholder values with your actual Google Cloud project ID:
    ```python
    # Google Cloud project ID
    PROJECT_ID = "your-google-cloud-project-id"
    LOCATION = "your-google-cloud-project-location"
    ```
    *(Replace `"your-google-cloud-project-id"` with your actual Google Cloud project ID and `"your-google-cloud-project-location"` with your actual Google Cloud location)*

## Running the Project

To execute the main script of the project, use the following command:

```bash
python3 src/machine_learning_base/main.py src/machine_learning_base/test.txt
```

By default, the main function use the heterogeneous dataset pre-trained model. You can 
change the set of pre-trained models by modifying main function in the `src/machine_learning_base/main.py` file.
Simply use the name of the directory containing the pre-trained models you want to use.
- all_text (This is called heterogeneous dataset in the research paper)
- Gemini_FullAI
- gemini_improved
- openAI_FullAI
- openAI_improved


# Running the Project with Data used for the Thesis
To execute any of the jupyter notebooks, you need to do the following:
1. **Upload the data files to MongoDB Atlas**
2. **Change the src/API/constant.py file to include the MongoDB Atlas collection names**
3. **Export your mongo password as an environment variable**
    ```bash
    export MONGODB_ATLAS_PASSWORD="your_mongo_password_here"
    ```
4. **Change the src/API/mongo_utils.py file to include the MongoDB Atlas connection string**
    ```python
    def get_mongo_client(username="your_username", cluster_url='cluster1.ha6ntos.mongodb.net', db_name='your_db_name',
                     password_env_var='MONGODB_ATLAS_PASSWORD'):
    ```
