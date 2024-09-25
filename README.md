# Fake News Detection System

_A machine learning project that detects and flags fake news articles by analyzing linguistic features and content patterns. The system is built using an LSTM (Long Short-Term Memory) neural network for text classification and deployed as a web application using Flask._

## Overview

In today's digital age, misinformation and fake news have become significant challenges. This project aims to address this issue by building a Fake News Detection System that classifies news articles as real or fake based on their textual content.

The project involves:

- Data preprocessing and text cleaning.
- Training an LSTM neural network model for classification.
- Developing a Flask web application for user interaction.

## Features

- **Accurate Classification**: Uses an LSTM model trained on a large dataset for high accuracy.
- **User-Friendly Interface**: Provides a simple web interface where users can input news articles.
- **Real-Time Detection**: Quickly processes and classifies input text.
- **Extensible**: The model and application can be extended to include more features or improved models.

## Demo

![Fake News Detection System Demo](demo_screenshot.png)

_Note: Replace the above image link with an actual screenshot or remove this section if not applicable._

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
├── static
│   └── styles.css
├── README.md
└── dataset
    ├── True.csv
    └── Fake.csv
