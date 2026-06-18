# FOMC Market Reaction Predictor

This project explores whether Federal Open Market Committee (FOMC) statement text can be used to predict short-term market reaction direction.

The workflow covers the full data science process: scraping FOMC statements from the Federal Reserve website, cleaning the statement text, combining the text with market reaction data, creating text-based features, training a machine learning model, and visualizing the results.

## Project Overview

The project:

- Scrapes FOMC statement text from the Federal Reserve website
- Uses Beautiful Soup to extract statement content from web pages
- Cleans and preprocesses statement language
- Combines statement text with market reaction labels
- Builds text-based features using TF-IDF and word frequency scores
- Trains a machine learning model to classify market reaction as `Up` or `Down`
- Saves trained model artifacts for reuse

## Data Collection and Web Scraping

FOMC statements were collected from the Federal Reserve website. The scraping workflow uses Beautiful Soup to parse the web pages and extract the statement text needed for analysis.

After scraping, the statement text is cleaned and structured into CSV files so it can be combined with market reaction data. This creates the dataset used for feature engineering and model training.

## Files

Key files in this project include:

- `predictive_model.py` - main Python script for training the prediction model
- `requirements.txt` - Python package requirements
- `scraping_speeches.ipynb` - notebook for collecting FOMC statement text from the Federal Reserve website
- `cleaned_speech_text.ipynb` - notebook for cleaning scraped statement text
- `merging_data.ipynb` - notebook for combining FOMC statements with market reaction data
- `frequency_analysis.ipynb` - notebook for analyzing word frequencies in FOMC statements
- `merged_fomc_market_data.csv` - merged statement and market reaction dataset
- `word_frequencies_with_market_reaction.csv` - word frequency features split by market reaction
- `fomc_predictor.joblib` - saved trained classifier
- `fomc_tfidf.joblib` - saved TF-IDF vectorizer
- `prediction_results.csv` - model prediction results
- `market_reaction_comparison.png` - visualization of predicted vs. actual market reactions
- `top_20_words.png` - visualization of common FOMC statement words

The notebooks show the project development process, including web scraping, text cleaning, data merging, frequency analysis, and model testing.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Model

From the project folder, run:

```bash
python predictive_model.py
```

The script loads `merged_fomc_market_data.csv`, trains the model, prints evaluation metrics, and saves the model files.

## Methods Used

- Text cleaning and tokenization with NLTK
- Lemmatization and stopword removal
- Web scraping with Beautiful Soup
- TF-IDF vectorization
- Word frequency scoring by market reaction
- Random forest classification
- Train/test evaluation with accuracy, classification report, and confusion matrix

## Notes

This project is for educational and research purposes. It is not financial advice and should not be used as the sole basis for investment or trading decisions.
