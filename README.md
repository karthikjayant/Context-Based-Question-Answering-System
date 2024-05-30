# Context-Based-Question-Answering-System
This project aims to create a Question Answering System using End-to-End Memory Networks. It starts by importing necessary libraries, including Keras for neural network construction and training, and utilities for data preprocessing and visualization. The dataset used is the bAbI tasks dataset, which is downloaded and extracted.

First, helper functions are defined for tokenizing sentences and parsing stories from the dataset. The tokenize function splits sentences into words, while the parse_stories function processes lines from the dataset into story-question-answer tuples. The get_stories function retrieves and processes these tuples from the dataset files, creating lists of stories, questions, and answers.

Next, the dataset is preprocessed. A vocabulary is built from the unique words in the stories and questions, and the maximum lengths of stories and questions are determined. These are used to vectorize the data, converting words to indices using the vocabulary. The vectorize_stories function pads sequences to the maximum lengths and creates binary labels for the answers.

The model architecture is then defined. Input sequences for stories and questions are created using Keras' Input layer. Embedding layers are used to encode these sequences into dense vector representations. Two separate encoders process the input stories, and one encoder processes the questions. A dot product is used to compute the match between story and question vectors, followed by a softmax activation to highlight relevant parts of the story.

The match matrix is combined with one of the story encoders and permuted to align with the question vectors. These are concatenated, and an LSTM layer processes the combined vectors. The output is passed through a dense layer and a softmax activation to generate a probability distribution over the vocabulary, predicting the answer.

The model is compiled with RMSprop optimizer and categorical crossentropy loss, and a summary of the model architecture is printed. The model is trained using the vectorized training data, with validation on the test data. Training progress is visualized using Keras callbacks.

After training, the model's performance is evaluated on test data. Predictions are made for a sample of test stories, and these predictions are compared to the ground truth answers to assess accuracy.

Finally, the trained model is saved for future use, allowing for the question-answering system to be deployed or further refined.
