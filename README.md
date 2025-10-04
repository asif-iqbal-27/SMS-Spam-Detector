# Spam Message Classification

This project aims to classify SMS messages as either "ham" (non-spam) or "spam" using machine learning techniques. The project is built using Python and several libraries such as Pandas, Scikit-Learn, NLTK, Streamlit, and XGBoost. The end result is a web application where users can input a message and classify it as either spam or ham.

## Project Overview
This project is designed to classify SMS messages as spam or ham. The steps include:
1. Data cleaning and preprocessing
2. Exploratory Data Analysis (EDA)
3. Text preprocessing including tokenization, stemming, and removal of stop words
4. Model building with various machine learning algorithms like Naive Bayes, Logistic Regression, SVM, Random Forest, etc.
5. Evaluation of the models' performance
6. Deployment of the model using Streamlit for frontend development

## Technologies Used
- **Python**: Core programming language
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning algorithms
- **NLTK**: Natural Language Processing (NLP)
- **XGBoost**: Enhanced Gradient Boosting Classifier
- **Streamlit**: For frontend development
- **Matplotlib, Seaborn**: For data visualization

## Features
- **Data Cleaning**: Removal of unnecessary columns, handling of duplicates, and missing values.
- **Exploratory Data Analysis (EDA)**: Visualizations to explore the data distribution, word count, character count, etc.
- **Text Preprocessing**: 
  - Converting text to lowercase.
  - Tokenizing text into words.
  - Removing stop words and special characters.
  - Stemming the words using the PorterStemmer.
  
After preprocessing, the text data was vectorized using **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical data suitable for machine learning models.

## Dataset
The dataset used is a labeled SMS dataset containing both ham (non-spam) and spam messages. It consists of the following columns:
- **target**: 0 for ham and 1 for spam.
- **text**: The actual message text.

The dataset was preprocessed by removing unnecessary columns, handling missing values, and performing text transformations.

## Data Preprocessing
- **Removing Unwanted Columns**: We dropped columns that contained too many missing values or irrelevant data.
- **Text Transformations**: 
  - Converting the text to lowercase.
  - Tokenizing text into words.
  - Removing stop words and special characters.
  - Stemming the words using the PorterStemmer.
  
After preprocessing, the text data was vectorized using **TF-IDF**.

## Model Building
Various machine learning algorithms were used for classification:
- **Naive Bayes (MultinomialNB, BernoulliNB)**
- **Support Vector Machine (SVC)**
- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost**
- **K-Nearest Neighbors (KNN)**
- **AdaBoost Classifier**

Performance metrics used to evaluate the models include **accuracy**, **precision**, **recall**, and **confusion matrix**.

## Frontend
The model is deployed with **Streamlit** for interactive user input. Users can input their SMS message into a text box, and the model will classify the message as spam or ham. The frontend also displays visualizations of model performance and accuracy.

## How to Run
1. **Clone the repository** and set up the environment.
2. **Install required libraries** through a Python environment or package manager.
3. **Run the Streamlit application** to interact with the model via a browser-based interface.

### Model and Vectorizer
- The trained model and the TF-IDF vectorizer are saved using **pickle** for future use.

## Evaluation
The following evaluation metrics were calculated for the models:
- **Accuracy**: The proportion of correct predictions (spam vs. ham).
- **Precision**: The ability of the model to correctly identify spam messages.
- **Confusion Matrix**: Helps visualize the performance of classification models.

The **Multinomial Naive Bayes (MNB)** model gave the best performance with an accuracy of **97.19%** and precision of **1.0**.

## Conclusion
This project demonstrates a comprehensive pipeline for text classification with machine learning, starting from data cleaning to model deployment. The classification model performs well with a variety of machine learning algorithms, with Naive Bayes being the best performing algorithm. The web application made with Streamlit allows users to interact with the model and classify new SMS messages in real time.

## Future Improvements
- Handle more complex imbalances in the dataset.
- Use more advanced deep learning models such as LSTM or BERT for improved accuracy.
- Enhance the frontend for better user interaction and experience.

