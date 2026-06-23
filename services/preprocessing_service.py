import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except LookupError:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def preprocess_text(text: str) -> dict:
    normalized = normalize_text(text)
    
    try:
        tokens = word_tokenize(normalized)
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
        tokens = word_tokenize(normalized)
        
    tokens_no_stop = [t for t in tokens if t not in stop_words and len(t) > 1]
    final_tokens = [lemmatizer.lemmatize(t) for t in tokens_no_stop]
        
    return {
        "original": text,
        "normalized": normalized,
        "tokens": tokens,
        "tokens_no_stop": tokens_no_stop,
        "final_tokens": final_tokens,
        "final_text": " ".join(final_tokens)
    }


def preprocess_batch(texts: list) -> list:
    return [preprocess_text(text)["final_text"] for text in texts]