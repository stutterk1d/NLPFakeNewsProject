# Fake News Detection System

_A project that detects and flags fake news by analyzing linguistic patterns, using an LSTM neural network for text classification, and deploying it via Flask as a web app._

## Overview

The project involves:

- Data preprocessing and text cleaning.
- Training an LSTM neural network model for classification.
- Developing a Flask web application for user interaction.

## Features

- Uses an LSTM model trained on a large dataset for high accuracy.
- Provides a simple web interface where users can input news articles.
- Quickly processes and classifies input text.
- The model and application can be extended to include more features or improved models.

![image](https://github.com/user-attachments/assets/9dfe0493-bfbe-4920-8237-1c5719b464c1)
![image](https://github.com/user-attachments/assets/43b5cfbb-242e-44a4-b869-9201e9e418c6)
![image](https://github.com/user-attachments/assets/376b9821-aeff-4a13-b1f0-c156c5c346a4)

## Project Structure

```plaintext
.
├── app.py
├── model_lstm.h5
├── tokenizer.pickle
├── requirements.txt
├── templates
│   ├── index.html
│   └── result.html
├── README.md
```

## Project Structure

- `app.py`: Main Flask application file.
- `model_lstm.h5`: Trained LSTM model file.
- `tokenizer.pickle`: Tokenizer used for text preprocessing.
- `requirements.txt`: Python dependencies.
- `templates/`: HTML templates for the web application.
- `static/`: Static files like CSS.
- `README.md`: Project documentation.

## Installation

### Prerequisites

- Python 3.6 or higher
- `pip` (Python package installer)
- Git (optional)

### Steps

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/NLPFakeNewsProject.git
    cd NLPFakeNewsProject
    ```

2. **Create a Virtual Environment**

    It's recommended to use a virtual environment to manage dependencies.

    ```bash
    python -m venv venv
    ```

3. **Activate the Virtual Environment**

    - On Windows:

    ```bash
    venv\Scripts\activate
    ```

    - On macOS/Linux:

    ```bash
    source venv/bin/activate
    ```

4. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

5. **Download NLTK Resources**

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

    _You can run the above commands in a Python shell._

6. **Place Model Files**

    Ensure `model_lstm.h5` and `tokenizer.pickle` are in the project root directory.

## Usage

1. **Run the Flask Application**

    ```bash
    python app.py
    ```

2. **Access the Application**

    Open your web browser and navigate to:

    [http://localhost:5000](http://localhost:5000)

3. **Classify News Articles**

    - Paste a news article into the text area.
    - Click the "Detect" button.
    - View the classification result on the next page.

## Dataset

The dataset used for training the model is the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle.

- **True.csv**: Contains real news articles.
- **Fake.csv**: Contains fake news articles.

_Note: Due to size constraints, the dataset files are not included in this repository.
## Model Training

The model is an LSTM neural network trained to classify news articles.

### Steps

1. **Data Preprocessing**

    - Combine `title` and `text` columns.
    - Clean text by removing URLs, punctuation, numbers, and extra whitespace.
    - Tokenize and remove stopwords.
    - Convert text to sequences using a tokenizer.
    - Pad sequences to ensure uniform input length.

2. **Model Architecture**

    - **Embedding Layer**: Converts words into embeddings.
    - **LSTM Layers**: Captures sequential dependencies in text.
    - **Dense Layer**: Outputs a probability between 0 and 1.

3. **Training**

    - Compiled with `binary_crossentropy` loss and `adam` optimizer.
    - Trained for multiple epochs with a validation split.

4. **Saving the Model**

    - Model saved as `model_lstm.h5`.
    - Tokenizer saved as `tokenizer.pickle`.

_For detailed code and steps, refer to the `NLP Fake News Project.ipynb` notebook (not included in this repository)._

## Web Application

The Flask web application provides a user interface for the Fake News Detection System.

### Key Files

- `app.py`: Contains the Flask application code.
- `templates/index.html`: Home page with a form to input news text.
- `templates/result.html`: Displays the classification result.

### Routes

- `/`: Renders the home page.
- `/predict`: Handles form submission and displays the result.

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
    - Flask
    - TensorFlow and Keras
    - NLTK
    - NumPy
    - Pandas
- **Frontend**:
    - HTML


## License

This project is licensed under the [MIT License](LICENSE).

## Contact

- **Name**: Ethan Tran
- **Email**: [ethantran03@gmail.com](mailto:ethantran03@gmail.com)
- **GitHub**: [stutterk1d](https://github.com/stutterk1d)

_Feel free to reach out if you have any questions or suggestions!_

