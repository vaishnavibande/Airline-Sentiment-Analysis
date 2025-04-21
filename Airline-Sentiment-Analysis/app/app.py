import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

# Initialize NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

# Load model
try:
    sentiment_model = joblib.load('logistic_regression_sentiment_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
except Exception as e:
    st.error(f"Error loading model files: {str(e)}")
    st.stop()

# Enhanced sentiment rules
CONTRAST_WORDS = ['but', 'however', 'although', 'except', 'yet', 'though']
POSITIVE_WORDS = ['good', 'great', 'excellent', 'awesome', 'comfortable', 'pleasant']
NEGATIVE_WORDS = ['bad', 'terrible', 'uncomfortable', 'poor', 'horrible', 'awful']
NEGATION_PHRASES = ['not good', 'not comfortable', "can't", 'cannot', 'unable to', 'failed to', 'was not', 'were not']
CONDITIONAL_WORDS = ['should', 'would', 'could', 'might', 'may']

def analyze_clauses(text):
    text_lower = text.lower()
    clauses = []
    current_clause = []
    
    # Split into clauses based on contrast words
    for word in text_lower.split():
        if word in CONTRAST_WORDS and current_clause:
            clauses.append(' '.join(current_clause))
            current_clause = []
        current_clause.append(word)
    if current_clause:
        clauses.append(' '.join(current_clause))
    
    return clauses

def contains_conditional(text):
    return any(word in text.lower() for word in CONDITIONAL_WORDS)

def contains_strong_negation(text):
    text_lower = text.lower()
    # Check for explicit negative phrases
    if any(phrase in text_lower for phrase in NEGATION_PHRASES):
        return True
    # Check for negative words
    if any(word in text_lower for word in NEGATIVE_WORDS):
        return True
    return False

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = [word for word in text.split() if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def determine_sentiment(text, raw_prediction):
    text_lower = text.lower()
    
    # First check for strong negative indicators
    if contains_strong_negation(text):
        return 'negative'
    
    # Check for conditional statements
    if contains_conditional(text):
        return 'neutral'
    
    # Check for contrast words
    if any(word in text_lower for word in CONTRAST_WORDS):
        clauses = analyze_clauses(text)
        
        # If we have multiple clauses with different sentiments
        if len(clauses) > 1:
            return 'neutral'
    
    return raw_prediction

# Streamlit UI
st.set_page_config(page_title="Airline Sentiment Classifier", layout="centered")
st.title("✈️ Airline Tweet Sentiment Classifier")

user_input = st.text_input("Enter Tweet:", "flight was not good")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            try:
                # Preprocess
                processed_text = preprocess(user_input)
                
                # Vectorize and predict
                vectorized = tfidf_vectorizer.transform([processed_text])
                raw_prediction = sentiment_model.predict(vectorized)[0]
                
                # Apply our rules
                final_prediction = determine_sentiment(user_input, raw_prediction)
                
                # Display results
                st.subheader("Original Text")
                st.write(user_input)
                
                st.subheader("Analysis")
                if final_prediction == "positive":
                    st.success("✅ Positive sentiment")
                elif final_prediction == "negative":
                    st.error("❌ Negative sentiment")
                else:
                    st.warning("⚠️ Neutral sentiment (mixed/conditional feedback)")
                
                st.subheader("Processing Details")
                with st.expander("Show details"):
                    st.write(f"Preprocessed text: `{processed_text}`")
                    st.write(f"Initial model prediction: {raw_prediction}")
                    
                    if contains_strong_negation(user_input):
                        st.write("**Negative indicators detected:**")
                        neg_phrases = [p for p in NEGATION_PHRASES if p in user_input.lower()]
                        neg_words = [w for w in NEGATIVE_WORDS if w in user_input.lower()]
                        if neg_phrases:
                            st.write(f"Negation phrases: {neg_phrases}")
                        if neg_words:
                            st.write(f"Negative words: {neg_words}")
                    
                    if any(word in user_input.lower() for word in CONTRAST_WORDS):
                        clauses = analyze_clauses(user_input)
                        st.write("**Clauses detected:**")
                        for clause in clauses:
                            st.write(f"- {clause}")
                    
                    if contains_conditional(user_input):
                        st.write("**Conditional words detected:**")
                        st.write([w for w in CONDITIONAL_WORDS if w in user_input.lower()])
                    
                    st.write(f"Final determination: {final_prediction}")
                    
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    else:
        st.warning("Please enter a tweet to analyze")

