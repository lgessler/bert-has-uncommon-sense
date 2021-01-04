# ensure we have semcor downloaded
import nltk
try:
    nltk.data.find('corpora/semcor.zip')
except LookupError:
    nltk.download('semcor')
