import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class FOMCPredictor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=100)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def prepare_features(self, df, fit_tfidf=False):
        # Text features
        processed_texts = df['clean_text'].apply(self.preprocess_text)
        if fit_tfidf:
            text_features = self.tfidf.fit_transform(processed_texts)
        else:
            text_features = self.tfidf.transform(processed_texts)
        
        # Additional features
        speech_length = df['speech_length'].values.reshape(-1, 1)
        
        # Load word frequencies
        word_freq_df = pd.read_csv('word_frequencies_with_market_reaction.csv', index_col='word')
        
        # Calculate frequency-based features
        freq_features = []
        for text in processed_texts:
            words = text.split()
            up_score = sum(word_freq_df.loc[word]['up_frequency'] 
                          for word in words if word in word_freq_df.index)
            down_score = sum(word_freq_df.loc[word]['down_frequency'] 
                           for word in words if word in word_freq_df.index)
            freq_features.append([up_score, down_score])
        
        freq_features = np.array(freq_features)
        
        # Combine all features
        return np.hstack([text_features.toarray(), speech_length, freq_features])
    
    def train(self, df):
        # Prepare features
        X = self.prepare_features(df, fit_tfidf=True)
        y = (df['market_reaction_up_or_down'] == 'Up').astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        y_pred = self.model.predict(X_test)
        
        print("Training accuracy:", train_score)
        print("Testing accuracy:", test_score)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save the model
        joblib.dump(self.model, 'fomc_predictor.joblib')
        joblib.dump(self.tfidf, 'fomc_tfidf.joblib')
    
    def predict(self, text, speech_length):
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Prepare features
        text_features = self.tfidf.transform([processed_text])
        speech_length = np.array([[speech_length]])
        
        # Load word frequencies
        word_freq_df = pd.read_csv('word_frequencies_with_market_reaction.csv', index_col='word')
        
        # Calculate frequency-based features
        words = processed_text.split()
        up_score = sum(word_freq_df.loc[word]['up_frequency'] 
                      for word in words if word in word_freq_df.index)
        down_score = sum(word_freq_df.loc[word]['down_frequency'] 
                        for word in words if word in word_freq_df.index)
        freq_features = np.array([[up_score, down_score]])
        
        # Combine features
        X = np.hstack([text_features.toarray(), speech_length, freq_features])
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        return 'Up' if prediction == 1 else 'Down', probabilities

if __name__ == '__main__':
    # Load data
    df = pd.read_csv('merged_fomc_market_data.csv')
    
    # Train model
    predictor = FOMCPredictor()
    predictor.train(df)
    print("\nModel training complete and saved!")
