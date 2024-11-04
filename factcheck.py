# factcheck.py


import torch
from typing import List
import numpy as np
import spacy
import gc



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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

    def check_entailment(self, premise: str, hypothesis: str) -> float:
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        # Return the probability of the "entailment" class
        entailment_prob = probs[0]  # Assuming 'entailment' is at index 0
        return entailment_prob

    def cleanup(self):
        # Explicitly delete large objects to save memory
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

import re
from typing import List

class EntailmentFactChecker:
    def __init__(self, ent_model: EntailmentModel, entailment_threshold: float = 0.6, overlap_threshold: float = 0.1):
        """
        :param ent_model: The entailment model for checking entailment
        :param entailment_threshold: Threshold to classify entailment probability as supported
        :param overlap_threshold: Optional word overlap threshold to filter unlikely passages
        """
        self.ent_model = ent_model
        self.entailment_threshold = entailment_threshold
        self.overlap_threshold = overlap_threshold

    def preprocess_text(self, text: str) -> List[str]:
        """
        Splits text into sentences and performs basic cleaning.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def word_overlap(self, fact: str, passage: str) -> float:
        """
        Computes word overlap as a rough filter to avoid processing irrelevant passages.
        """
        fact_tokens = set(fact.lower().split())
        passage_tokens = set(passage.lower().split())
        overlap = len(fact_tokens.intersection(passage_tokens)) / len(fact_tokens) if len(fact_tokens) > 0 else 0
        return overlap

    def predict(self, fact: str, passages: List[dict]) -> str:
        max_entailment_score = 0

        # Loop over all passages
        for passage in passages:
            passage_text = passage['text']
            sentences = self.preprocess_text(passage_text)

            # Prune passages with low word overlap
            if self.word_overlap(fact, passage_text) < self.overlap_threshold:
                continue

            # Check each sentence in the passage
            for sentence in sentences:
                entailment_prob = self.ent_model.check_entailment(sentence, fact)
                max_entailment_score = max(max_entailment_score, entailment_prob)

        # Decide supported vs. not supported based on the highest entailment score
        return "S" if max_entailment_score >= self.entailment_threshold else "NS"


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



class FactChecker:
    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Don't call me, call my subclasses")

class WordRecallThresholdFactChecker:
    def __init__(self, threshold=0.5, bigram_weight=.5):
        """
        :param threshold: The similarity score threshold for classification
        :param bigram_weight: Weight applied to bigram overlap to increase its influence
        """
        self.threshold = threshold
        self.bigram_weight = bigram_weight

    def preprocess_text(self, text):
        """
        Preprocesses the text by:
        - Converting to lowercase
        - Removing punctuation
        - Tokenizing words
        - Adding bigrams for more context
        """
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', '', text.lower())

        # Tokenize words
        tokens = text.split()

        # Generate bigrams for additional context
        bigrams = [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]

        # Return both unigrams and bigrams
        return tokens + bigrams

    def compute_overlap_score(self, fact, passage):
        # Preprocess and tokenize both fact and passage
        fact_tokens = set(self.preprocess_text(fact))
        passage_tokens = set(self.preprocess_text(passage))

        # Separate unigrams and bigrams for custom weighting
        fact_unigrams = {token for token in fact_tokens if "_" not in token}
        fact_bigrams = fact_tokens - fact_unigrams
        passage_unigrams = {token for token in passage_tokens if "_" not in token}
        passage_bigrams = passage_tokens - passage_unigrams

        # Calculate overlap scores with separate weights
        unigram_overlap = len(fact_unigrams.intersection(passage_unigrams)) / min(len(fact_unigrams),
                                                                                  len(passage_unigrams)) if fact_unigrams and passage_unigrams else 0
        bigram_overlap = len(fact_bigrams.intersection(passage_bigrams)) / min(len(fact_bigrams),
                                                                               len(passage_bigrams)) if fact_bigrams and passage_bigrams else 0

        # Combined score using bigram weight
        combined_score = (unigram_overlap + self.bigram_weight * bigram_overlap) / (1 + self.bigram_weight)
        return combined_score

    def predict(self, fact, passages):
        max_overlap_score = 0

        if isinstance(passages, list):  # Handle multiple passages
            max_overlap_score = max(self.compute_overlap_score(fact, passage['text']) for passage in passages)
            return "S" if max_overlap_score >= self.threshold else "NS"
        else:  # Handle single passage text
            overlap_score = self.compute_overlap_score(fact, passages)
            return overlap_score >= self.threshold


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
