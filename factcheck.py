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
from transformers import AutoTokenizer, AutoModel

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
class WordRecallThresholdFactChecker(FactChecker):
    def __init__(self, threshold=0.1, alpha=0.3, beta=0.3, model_name='bert-base-uncased', all_facts=None, all_passages=None):
        """
        :param threshold: The similarity score threshold for classification
        :param alpha: Weighting factor for combining cosine and Jaccard similarity
        :param beta: Weight for incorporating BLEU score
        :param model_name: Name of the pre-trained BERT model to use
        """
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.nlp = spacy.load('en_core_web_sm')

        # Load BERT tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Ensure all_facts and all_passages are provided for building the vocabulary
        if all_facts is not None and all_passages is not None:
            flat_passages = [text for sublist in all_passages for text in sublist]
            self.build_vocabulary(all_facts, flat_passages)
        else:
            raise ValueError("all_facts and all_passages must be provided to build vocabulary")

    def custom_tokenizer(self, text):
        doc = self.nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
        return tokens

    def build_vocabulary(self, all_facts, flat_passages):
        # Vocabulary is not needed for BERT-based embeddings but can be implemented if necessary
        pass

    def compute_bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the mean of the last hidden state as the embedding
        return outputs.last_hidden_state.mean(dim=1)

    def compute_overlap_score(self, fact, passage):
        # Generate embeddings for fact and passage
        fact_embedding = self.compute_bert_embedding(fact)
        passage_embedding = self.compute_bert_embedding(passage)
        # Cosine similarity between the embeddings
        return cosine_similarity(fact_embedding, passage_embedding)[0][0]

    def compute_jaccard_similarity(self, fact, passage):
        # Jaccard similarity using token sets
        fact_tokens = set(self.custom_tokenizer(fact))
        passage_tokens = set(self.custom_tokenizer(passage))
        intersection = fact_tokens.intersection(passage_tokens)
        union = fact_tokens.union(passage_tokens)
        return len(intersection) / len(union) if union else 0

    def compute_bleu_score(self, fact, passage):
        # BLEU score calculation with smoothing
        fact_tokens = self.custom_tokenizer(fact)
        passage_tokens = self.custom_tokenizer(passage)
        smoothing_function = SmoothingFunction().method1  # Choose the appropriate smoothing method
        return sentence_bleu([passage_tokens], fact_tokens, smoothing_function=smoothing_function)

    def predict(self, fact, passages):
        max_score = 0

        for passage in passages:
            passage_text = passage['text']

            # Calculate each similarity score
            cosine_score = self.compute_overlap_score(fact, passage_text)
            jaccard_score = self.compute_jaccard_similarity(fact, passage_text)
            bleu_score = self.compute_bleu_score(fact, passage_text)

            # Combined hybrid score with weighted components
            hybrid_score = (self.alpha * cosine_score + (1 - self.alpha) * jaccard_score) * (1 + self.beta * bleu_score)
            max_score = max(max_score, hybrid_score)

        # Classify based on threshold
        return "S" if max_score >= self.threshold else "NS"

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
