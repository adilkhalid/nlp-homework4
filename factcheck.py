# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc

import re

from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import torch
from typing import List
import numpy as np
import spacy
import gc
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.ja.syntax_iterators import labels
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # If the model is not found, download it
    from spacy.cli import download

    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')


class FactExample:
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """

    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        raise Exception("Not implemented")

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()

        # return something


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import spacy
import numpy as np

from sklearn.linear_model import LogisticRegression
from nltk.translate.bleu_score import sentence_bleu
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy


import torch
from typing import List
import numpy as np
import spacy
import gc
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class FactChecker:
    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Don't call me, call my subclasses")


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from nltk.corpus import stopwords
from itertools import chain, tee


class WordRecallThresholdFactChecker:
    def __init__(self, threshold=0.5, include_bigrams=True):
        """
        :param threshold: The similarity score threshold for classification
        :param include_bigrams: If True, includes bigrams in tokenization for more context
        """
        self.threshold = threshold
        self.include_bigrams = include_bigrams

    def preprocess_text(self, text):
        """
        Preprocesses the text by:
        - Converting to lowercase
        - Removing punctuation
        - Stripping extra whitespace
        - Tokenizing words
        - Optionally adding bigrams
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)

        # Tokenize words
        tokens = text.split()

        # Optionally add bigrams to tokens list
        if self.include_bigrams and len(tokens) > 1:
            bigrams = [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
            tokens.extend(bigrams)

        return tokens

    def compute_overlap_score(self, fact, passage):
        # Preprocess and tokenize both fact and passage
        fact_tokens = set(self.preprocess_text(fact))
        passage_tokens = set(self.preprocess_text(passage))

        # Avoid division by zero if tokens are empty
        if not fact_tokens or not passage_tokens:
            return 0

        # Szymkiewicz–Simpson (Overlap) Coefficient
        return len(fact_tokens.intersection(passage_tokens)) / min(len(fact_tokens), len(passage_tokens))

    def predict(self, fact, passages):
        max_overlap_score = 0

        for passage in passages:
            passage_text = passage['text']
            # Calculate overlap score for each passage
            overlap_score = self.compute_overlap_score(fact, passage_text)
            max_overlap_score = max(max_overlap_score, overlap_score)

        # Classify based on overlap score threshold
        return "S" if max_overlap_score >= self.threshold else "NS"

class EntailmentFactChecker(object):
    def __init__(self, ent_model):
        self.ent_model = ent_model

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")


# OPTIONAL
class DependencyRecallThresholdFactChecker(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations
