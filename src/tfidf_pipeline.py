from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

import config


def build_pipeline(c_value, word_ngram_max, class_weight):
    # Shared by the tuning script and the final test script, so both build the same model.
    # Word n-grams catch phrases like "iyi değil".
    word_tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, word_ngram_max), min_df=2)
    # Character n-grams (inside word borders) catch Turkish suffixes and typos:
    # "güzel", "güzeldi", "güzelmiş" share most of their character pieces.
    char_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    features = FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])

    # max_iter is raised because the default (100) stops before converging on this many features.
    classifier = LogisticRegression(
        C=c_value,
        class_weight=class_weight,
        max_iter=1000,
        random_state=config.SEED,
    )
    return Pipeline([("features", features), ("classifier", classifier)])
