# Sentiment Analysis and Text Generation with LSTM Neural Networks

## Project Overview

This project delves into sentiment analysis and text generation using many-to-one Long Short-Term Memory (LSTM) neural networks. It consists of two main components:

### Sentiment Detection

We aim to analyze airline sentiments by training an LSTM model on a dataset containing sentiment labels (0 or 1) and corresponding text reviews. The process involves:

1. **Dataset**: Obtain the airline sentiment dataset.
2. **Preprocessing**: Clean the text, tokenize, and convert it into a bag-of-words representation.
3. **Many-to-One LSTM**: Utilize LSTM architecture for sentiment detection.
4. **Training**: Split the dataset, train the LSTM model, and evaluate its performance.

### Text Generation

Using "Alice's Adventures in Wonderland" as our training text, we will generate new text based on a given prompt. The steps include:

1. **Dataset**: Obtain the text dataset.
2. **Preparing and Structuring**: Clean, tokenize, and structure the text data.
3. **Many-to-One LSTM**: Implement LSTM architecture for text generation.
4. **Challenges**: Address challenges such as language variability and word choice.
5. **Entropy Scaling and Softmax Temperature**: Introduce techniques to control randomness in text generation.

## Approach

### Sentiment Analysis

#### Dataset
- Obtain the airline sentiment dataset.

#### Preprocessing
- Clean the text, tokenize, and remove stop words.
- Convert text reviews into a bag-of-words representation.

#### Many-to-One LSTM
- Use LSTM architecture for sentiment detection.
- Feed bag-of-words representation as input.

#### Training
- Split dataset into training and testing sets.
- Train the LSTM model and evaluate its performance.

### Text Generation

#### Dataset
- Obtain "Alice's Adventures in Wonderland" text dataset.

#### Preparing and Structuring
- Clean, tokenize, and structure the text data.
- Create sequences for training.

#### Many-to-One LSTM
- Implement LSTM architecture for text generation.
- Train the model using the prepared dataset.

#### Challenges and Techniques
- Understand challenges in text generation.
- Introduce entropy scaling and softmax temperature for controlling randomness.

## Important Libraries

- **TensorFlow**: For building and training neural networks.
- **NumPy**: For efficient mathematical operations on arrays.
- **Scikit-learn**: For machine learning algorithms and utilities.
- **Seaborn** and **Matplotlib**: For data visualization.
- **Pandas**: For data manipulation and analysis.
- **NLTK**: For natural language processing tasks.
