import torch
from typing import List
import numpy as np
import nltk
import gc

# Ensure necessary NLTK data packages are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

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
        entailment_prob = probs[0]  # Adjust the index if necessary based on your model
        return entailment_prob

    def cleanup(self):
        # Explicitly delete large objects to save memory
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

import re

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
        Splits text into sentences using NLTK's sent_tokenize.
        """
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
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
        # Initialize any required NLTK resources
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :return: A set of dependency relations as tuples (head, label, child)
        """
        # Tokenize and POS tag the sentence
        tokens = nltk.word_tokenize(sent)
        pos_tags = nltk.pos_tag(tokens)

        # For dependency parsing, NLTK doesn't have a built-in parser like spaCy
        # You might need to use a third-party parser or implement a simple one

        # Here, we'll create a simple placeholder that returns an empty set
        # Replace this with actual dependency parsing if needed
        relations = set()
        return relations
