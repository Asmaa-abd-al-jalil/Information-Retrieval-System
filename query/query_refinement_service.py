from spellchecker import SpellChecker

spell = SpellChecker()

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