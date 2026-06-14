from spellchecker import SpellChecker
from nltk.corpus import wordnet
from collections import Counter

spell = SpellChecker()
user_history = []

def correct_query(query):
    words = query.split()
    corrected_words = []
    for word in words:
        corrected = spell.correction(word)
        if corrected:
            corrected_words.append(corrected)
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)

def expand_query(query):
    expanded_terms = []
    for word in query.split():
        expanded_terms.append(word)
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                if synonym not in expanded_terms:
                    expanded_terms.append(synonym)
    return " ".join(expanded_terms[:10])

def add_to_history(query):
    user_history.append(query)

def weight_query_with_history(query):
    if not user_history:
        return query
    all_words = " ".join(user_history).split()
    top_terms = [word for word, _ in Counter(all_words).most_common(3)]
    query_words = query.split()
    extra = [t for t in top_terms if t not in query_words]
    return query + " " + " ".join(extra)

def refine_query(query):
    print("Original Query    :", query)
    corrected = correct_query(query)
    print("Corrected Query   :", corrected)
    expanded = expand_query(corrected)
    print("Expanded Query    :", expanded)
    weighted = weight_query_with_history(expanded)
    print("Weighted Query    :", weighted)
    add_to_history(query)
    print()
    return weighted

# اختبار
refine_query("machne learnng")
refine_query("informtion retrival")
refine_query("neutrl netwrk")