import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# ── تنزيل الموارد مرة وحدة ──
for _pkg in ("punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger",
             "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

stop_words  = set(stopwords.words("english"))
stemmer     = PorterStemmer()
lemmatizer  = WordNetLemmatizer()

# كلمات طبية شائعة جداً لا تضيف معنى في clinical trials
_MEDICAL_STOP = {
    "patient", "patients", "study", "clinical", "trial", "trials",
    "treatment", "disease", "condition", "group", "groups",
    "include", "exclude", "use", "used", "using",
}
stop_words |= _MEDICAL_STOP


def _get_wordnet_pos(treebank_tag: str) -> str:
    """تحويل POS tag لـ WordNet format."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # default


def normalize_text(text: str) -> str:
    """تنظيف النص: lowercase + إزالة الرموز + توحيد المسافات."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text: str, use_stemming: bool = False) -> dict:
    """
    Pipeline معالجة موحّد للوثائق والاستعلامات.

    الخطوات:
      1. Normalize (lowercase, remove symbols)
      2. Tokenize
      3. Remove stopwords (English + medical)
      4. Lemmatize (افتراضي) أو Stem (اختياري)

    Returns dict بنفس الـ keys في كل مكان:
      original     : النص الأصلي
      normalized   : بعد التنظيف
      tokens       : كل الـ tokens
      tokens_no_stop: بعد حذف stop words
      final_tokens : بعد lemmatization/stemming
      final_text   : final_tokens مجموعة بمسافات
    """
    normalized = normalize_text(text)
    tokens     = word_tokenize(normalized)

    # حذف stop words والكلمات القصيرة جداً
    tokens_no_stop = [
        t for t in tokens
        if t not in stop_words and len(t) > 1
    ]

    if use_stemming:
        final_tokens = [stemmer.stem(t) for t in tokens_no_stop]
    else:
        # Lemmatization مع POS tagging أدق
        tagged       = pos_tag(tokens_no_stop)
        final_tokens = [
            lemmatizer.lemmatize(word, _get_wordnet_pos(tag))
            for word, tag in tagged
        ]

    return {
        "original":      text,
        "normalized":    normalized,
        "tokens":        tokens,
        "tokens_no_stop": tokens_no_stop,
        "final_tokens":  final_tokens,
        "final_text":    " ".join(final_tokens),
    }
