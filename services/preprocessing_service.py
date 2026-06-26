import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

def normalize_text(text: str) -> str:
    """تنظيف النص الأساسي."""
    if not isinstance(text, str): 
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_text(text: str) -> dict:
    """
    المعالجة المسبقة للمستندات والاستعلامات.
    ترجع القاموس المطلوب ليتوافق مع نظام الاسترجاع لديك.
    """
    normalized = normalize_text(text)
    tokens = word_tokenize(normalized)
    
    final_tokens = [
        stemmer.stem(t) 
        for t in tokens 
        if t not in stop_words and len(t) > 1
    ]
    
    return {
        "original": text,
        "final_tokens": final_tokens,
        "final_text": " ".join(final_tokens)
    }