# rewards.py

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import torch
import numpy as np
from collections import Counter
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from readability.readability import Readability
import spacy
from typing import List, Dict, Tuple
import json

nlp = spacy.load('en_core_web_sm')

class RewardRegistry:
    registry = {}
    weights = {}
    min_scores = {}
    max_scores = {}

    @classmethod
    def register(cls, name):
        def decorator(func):
            cls.registry[name] = func
            cls.min_scores[name] = None
            cls.max_scores[name] = None
            cls._recalculate_weights()
            return func
        return decorator

    @classmethod
    def _recalculate_weights(cls):
        num_rewards = len(cls.registry)
        if num_rewards > 0:
            uniform_weight = 1 / num_rewards
            for key in cls.registry:
                cls.weights[key] = uniform_weight

    @classmethod
    def set_weights(cls, new_weights):
        for name, weight in new_weights.items():
            if name in cls.weights:
                cls.weights[name] = weight
            else:
                raise ValueError(f"No reward function registered under the name {name}.")

    @classmethod
    def update_score_range(cls, name, score):
        if cls.min_scores[name] is None or score < cls.min_scores[name]:
            cls.min_scores[name] = score
        if cls.max_scores[name] is None or score > cls.max_scores[name]:
            cls.max_scores[name] = score

    @classmethod
    def normalize_score(cls, name, score):
        if cls.min_scores[name] is not None and cls.max_scores[name] is not None:
            range = cls.max_scores[name] - cls.min_scores[name]
            if range > 0:
                return (score - cls.min_scores[name]) / range
            else:
                return 0.0
        return score

    @classmethod
    def compute_total_reward(cls, generated, references, weights, logits):
        total_rewards = torch.zeros(len(generated), device="cuda" if torch.cuda.is_available() else "cpu")
        
        for name, func in cls.registry.items():
            if name in weights:
                raw_scores = func(generated, references, logits)
                for raw_score in raw_scores:
                    cls.update_score_range(name, raw_score.item())
                normalized_scores = torch.tensor([cls.normalize_score(name, score.item()) for score in raw_scores], device="cuda" if torch.cuda.is_available() else "cpu")

        return normalize_scores

""" @RewardRegistry.register("cider")
def cider_reward(generated: List[str], references: List[str], logits: torch.Tensor) -> torch.Tensor:
    cider_scorer = Cider()
    image_ids = range(len(references))
    formatted_hypotheses = {image_id: [cap] for image_id, cap in zip(image_ids, generated)}
    formatted_references = {image_id: [cap] for image_id, cap in zip(image_ids, references)}
    score, _ = cider_scorer.compute_score(formatted_references, formatted_hypotheses)
    return torch.tensor([score] * len(generated)) """

""" @RewardRegistry.register("rouge")
def rouge_reward(generated: List[str], references: List[str], logits: torch.Tensor) -> torch.Tensor:
    rouge_scorer = Rouge()
    scores = []
    for gen, ref in zip(generated, references):
        formatted_hypotheses = {0: [gen]}
        formatted_references = {0: [ref]}
        score, _ = rouge_scorer.compute_score(formatted_references, formatted_hypotheses)
        scores.append(score)
    return torch.tensor(scores, device="cuda" if torch.cuda.is_available() else "cpu") """

""" @RewardRegistry.register("repetition_penalty")
def repetition_penalty_reward(generated: List[str], references: List[str], logits: torch.Tensor) -> torch.Tensor:
    def count_ngrams(words, n):
        ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        ngram_counts = Counter(ngrams)
        return sum(count - 1 for count in ngram_counts.values() if count > 1)

    penalties = []
    for caption in generated:
        words = caption.split()
        unigram_repetition = sum(count - 1 for count in Counter(words).values() if count > 1)
        bigram_repetition = count_ngrams(words, 2)
        trigram_repetition = count_ngrams(words, 3)
        fourgram_repetition = count_ngrams(words, 4)
        total_penalty = (unigram_repetition + 2 * bigram_repetition + 3 * trigram_repetition + 4 * fourgram_repetition)**2
        penalties.append(-total_penalty)
    return torch.tensor(penalties, device="cuda" if torch.cuda.is_available() else "cpu") """

""" class ReadabilityReward:
    def __init__(self):
        self.ih = 0

    def flesch_kincaid_grade(self, text: str) -> float:
        r = Readability(text)
        fk = r.flesch_kincaid()
        return fk.score

    def gunning_fog_index(self, text: str) -> float:
        r = Readability(text)
        gf = r.gunning_fog()
        return gf.score

    def compute_score(self, generated: List[str], references: List[str]) -> torch.Tensor:
        concatenated_generated = " ".join(generated)
        concatenated_references = " ".join(references)
        fk_gen = self.flesch_kincaid_grade(concatenated_generated)
        fk_ref = self.flesch_kincaid_grade(concatenated_references)
        gf_gen = self.gunning_fog_index(concatenated_generated)
        gf_ref = self.gunning_fog_index(concatenated_references)
        fk_gap = abs(fk_gen - fk_ref)
        gf_gap = abs(gf_gen - gf_ref)
        readability_score = fk_gap + gf_gap
        scores = [readability_score] * len(generated)
        return -torch.tensor(scores, device="cuda" if torch.cuda.is_available() else "cpu")

grammar_reward_scorer = ReadabilityReward()

@RewardRegistry.register("grammar")
def grammar_reward(generated: List[str], references: List[str], logits: torch.Tensor) -> torch.Tensor:
    return grammar_reward_scorer.compute_score(generated, references) """

""" @RewardRegistry.register("information_density")
def information_density_reward(generated: List[str], references: List[str], logits: torch.Tensor) -> torch.Tensor:
    scores = []
    for caption in generated:
        doc = nlp(caption)
        num_entities = len(doc.ents)
        num_words = len(caption.split())
        score = num_entities / num_words if num_words > 0 else 0
        scores.append(score)
    return torch.tensor(scores, device="cuda" if torch.cuda.is_available() else "cpu") """

""" @RewardRegistry.register("structural_complexity")
def structural_complexity_reward(generated: List[str], references: List[str], logits: torch.Tensor) -> torch.Tensor:
    scores = []
    for caption in generated:
        doc = nlp(caption)
        depth = max(token.head.i - token.i for token in doc)
        num_clauses = len([sent for sent in doc.sents])
        score = depth + num_clauses
        scores.append(score)
    return torch.tensor(scores, device="cuda" if torch.cuda.is_available() else "cpu") """

""" @RewardRegistry.register("meteor")
def meteor_reward(generated: List[str], references: List[str], logits: torch.Tensor) -> torch.Tensor:
    meteor_scorer = Meteor()
    scores = []
    for gen, ref in zip(generated, references):
        score, _ = meteor_scorer.compute_score({0: [ref]}, {0: [gen]})
        scores.append(score)
    return torch.tensor(scores, device="cuda" if torch.cuda.is_available() else "cpu") """

def calculate_proportion(gt_obj: List[List[str]], gen_obj: List[List[str]]) -> Tuple[List[float], List[float]]:
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    def lemmatize_list(words):
        return [lemmatizer.lemmatize(word.lower()) for word in words]

    proportions_gen_in_gt = []
    proportions_gt_not_in_gen = []

    for gt_list, gen_list in zip(gt_obj, gen_obj):
        lemmatized_gt = lemmatize_list(gt_list)
        lemmatized_gen = lemmatize_list(gen_list)

        match_count_gen_in_gt = sum(word in lemmatized_gt for word in lemmatized_gen)
        proportion_gen_in_gt = match_count_gen_in_gt / len(lemmatized_gen) if lemmatized_gen else 0
        proportions_gen_in_gt.append(proportion_gen_in_gt)

        match_count_gt_not_in_gen = sum(word not in lemmatized_gen for word in lemmatized_gt)
        proportion_gt_not_in_gen = match_count_gt_not_in_gen / len(lemmatized_gt) if lemmatized_gt else 0
        proportions_gt_not_in_gen.append(proportion_gt_not_in_gen)

    return proportions_gen_in_gt, proportions_gt_not_in_gen

""" @RewardRegistry.register("object_matching")
def object_matching_reward(generated: List[str], references: List[str], logits: torch.Tensor) -> torch.Tensor:
    from scenegraph import scenegraphmaker
    gen_obj = scenegraphmaker(generated)
    gt_obj = scenegraphmaker(references)
    
    prop_scores, hallu_prop_scores = calculate_proportion(gt_obj, gen_obj)
    
    combined_scores = [p - h for p, h in zip(prop_scores, hallu_prop_scores)]
    return torch.tensor(combined_scores, device="cuda" if torch.cuda.is_available() else "cpu") """

class RollingVocabKLDiv:
    def __init__(self, json_path: str, vocab_size: int = 10000, epsilon: float = 1e-7, update_freq: int = 100, smoothing_factor: float = 1.0):
        self.vocab_size = vocab_size
        self.epsilon = epsilon
        self.update_freq = update_freq
        self.smoothing_factor = smoothing_factor
        self.rolling_vocab_freq = Counter()
        self.gt_vocab_prob = self._get_word_frequencies(self._load_json(json_path))
        self.kl_div_loss: float = 0.
        self.temp_captions = []

    def _load_json(self, file_path):
        with open(file_path) as f:
            data = json.load(f)
        return data['annotations']

    def _get_word_frequencies(self, annotations):
        text_data = ' '.join([annotation['caption'] for annotation in annotations]).lower()
        words = word_tokenize(text_data)
        word_frequencies = Counter(words)
        total_words = sum(word_frequencies.values())
        return {word: freq / total_words for word, freq in word_frequencies.items()}

    def _batch_update(self):
        for caption in self.temp_captions:
            words = word_tokenize(caption.lower())
            for word in words:
                if len(self.rolling_vocab_freq) < self.vocab_size or word in self.rolling_vocab_freq:
                    self.rolling_vocab_freq[word] += 1
        self.temp_captions = []

    def update_rolling_vocab_distribution(self, new_captions):
        self.temp_captions.extend(new_captions)
        if len(self.temp_captions) >= self.update_freq:
            self._batch_update()

    def compute_kl_div_loss(self):
        if self.temp_captions:
            self._batch_update()

        total_words = sum(self.rolling_vocab_freq.values()) + self.smoothing_factor * len(self.gt_vocab_prob)
        rolling_vocab_prob = {word: (freq + self.smoothing_factor) / total_words for word, freq in self.rolling_vocab_freq.items()}

        kl_div = 0.0
        for word, prob in self.gt_vocab_prob.items():
            prob_rolling = rolling_vocab_prob.get(word, self.epsilon)
            kl_div += prob * np.log(prob / prob_rolling)

        self.kl_div_loss = kl_div

        return kl_div

@RewardRegistry.register("kl_divergence")
def kl_divergence_reward(generated: List[str], references: List[str], logits: torch.Tensor) -> torch.Tensor:
    kl_div_calculator = RollingVocabKLDiv('train2.json', vocab_size=10000)
    kl_div_calculator.update_rolling_vocab_distribution(generated)
    kl_div_loss = kl_div_calculator.compute_kl_div_loss()
    return torch.tensor([kl_div_loss] * len(generated), device="cuda" if torch.cuda.is_available() else "cpu")