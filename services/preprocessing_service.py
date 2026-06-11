import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# يجب التأكد من تحميل مكتبة nltk: 
# nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. تحويل لأحرف صغيرة
    text = text.lower()
    
    # 2. إزالة الرموز والحروف غير الأبجدية
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. التقسيم (Tokenization) وإزالة الكلمات الشائعة (Stopwords) والـ Stemming
    tokens = text.split()
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return " ".join(processed_tokens)