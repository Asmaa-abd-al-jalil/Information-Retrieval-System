import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# تحميل الموارد اللازمة
nltk.download('stopwords')

# إعداد الأدوات
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """دالة لمعالجة النصوص (Normalization + Tokenization + Stopwords + Stemming)"""
    if not isinstance(text, str):
        return ""
    
    # 1. Normalization: تحويل لأحرف صغيرة وإزالة الرموز
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # 2. Tokenization: تقسيم النص
    tokens = text.split()
    
    # 3. Stopwords & Stemming
    processed = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return " ".join(processed)