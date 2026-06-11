import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# تحميل الموارد
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# إعداد الأدوات
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def normalize(text: str) -> str:
    """تحويل النص لأحرف صغيرة وإزالة الرموز"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str) -> list:
    """تقسيم النص لكلمات"""
    return text.split()

def remove_stopwords(tokens: list) -> list:
    """إزالة الكلمات الشائعة"""
    return [t for t in tokens if t not in stop_words]

def stem(tokens: list) -> list:
    """Stemming للكلمات"""
    return [stemmer.stem(t) for t in tokens]

def lemmatize(tokens: list) -> list:
    """Lemmatization للكلمات"""
    return [lemmatizer.lemmatize(t) for t in tokens]

def preprocess_text(text: str, use_stemming: bool = True) -> dict:
    """
    الدالة الرئيسية للمعالجة - بترجع كل مرحلة بشكل منفصل
    use_stemming: True = Stemming, False = Lemmatization
    """
    # المراحل بالترتيب
    normalized = normalize(text)
    tokens = tokenize(normalized)
    tokens_no_stop = remove_stopwords(tokens)
    
    if use_stemming:
        final_tokens = stem(tokens_no_stop)
    else:
        final_tokens = lemmatize(tokens_no_stop)
    
    return {
        "original": text,
        "normalized": normalized,
        "tokens": tokens,
        "tokens_no_stop": tokens_no_stop,
        "final_tokens": final_tokens,           # هاد يلي رح يستخدمه العضو 2
        "final_text": " ".join(final_tokens)    # نص جاهز للـ TF-IDF
    }