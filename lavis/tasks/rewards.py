# rewards.py##GREEDY FORCING##GREEDY FORCING##GREEDY FORCING##GREEDY FORCING##GREEDY FORCING##GREEDY FORCING##GREEDY FORCING##GREEDY FORCING##GREEDY FORCING

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import torch
import numpy as np
from collections import Counter
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from readability.readability import Readability
import spacy
from transformers import CLIPTokenizer, CLIPModel
from scipy.spatial.distance import cosine
from typing import List, Dict, Tuple
from scipy.spatial.distance import cosine
import json

import time
import re

# Pipeline to use one of Spacy's small pretrained language model
nlp = spacy.load('en_core_web_sm')

# Preoad CLIP model and tokenizer (for get_clip_embedding)
model_name = "openai/clip-vit-base-patch32"
tokenizer = CLIPTokenizer.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Registry for the custom learning signals (rewards are fixed, not learned)
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

    # In case the importance weights given to each signal are modified
    @classmethod
    def _recalculate_weights(cls):
        num_rewards = len(cls.registry)
        if num_rewards > 0:
            uniform_weight = 1 / num_rewards
            for key in cls.registry:
                cls.weights[key] = uniform_weight

    @classmethod
    def set_weights(cls, new_weights):
        """

        Sets new weights (inplace) to each learning signal using the new_weights dictionary.

        Args:
            new_weights (dict): a dictionary of weights with the format {learning_signal_name: weight}
        Return:
            None
        """
        for name, weight in new_weights.items():
            if name in cls.weights:
                cls.weights[name] = weight
            else:
                raise ValueError(f"No reward function registered under the name {name}.")

    @classmethod
    def normalize_score(cls, name, raw_scores):
        """
        Normalizes each learning signal (normalize_score) 
        before computing the weighted sum of normalized learning signals (compute_total_reward)

        Args:
            name (str): the name of the learning signal to normalize.
            raw_scores (torch.Tensor): tensor containing the unnormalized learning signals.
        Return:
            A torch.Tensor with the same format as raw_scores that contains the normalized scores
        """
        raw_scores_np = raw_scores.cpu().numpy()
        if cls.min_scores[name] is None or cls.max_scores[name] is None:
            cls.min_scores[name] = raw_scores_np.min()
            cls.max_scores[name] = raw_scores_np.max()

        min_score, max_score = cls.min_scores[name], cls.max_scores[name]
        normalized = (raw_scores_np - min_score) / (max_score - min_score + 1e-8)
        return torch.tensor(normalized, device=raw_scores.device)

    @classmethod
    def compute_total_reward(cls, generated: List[str], references: List[str], weights: dict, logits: torch.Tensor):
    """

    Computation of the reward for SCST as the weighted sum of the normalized learning signals

    Args:
        generated (List[str)]): a batch of strings corresponding to the generated captions.
        references (List[str)]): a batch of reference captions, aligned with "generated".
        weights (dict): a dictionary containing the weights of each learning signal in the format {name: weights}.
        logits (torch.Tensor): the logits output by the LLM, in the format (batch * num_tokens * vocab_size).
    Return:
        total_rewards (float): the weighted sum of normalized learning signals.
    """
        total_rewards = torch.zeros(len(generated), device="cuda" if torch.cuda.is_available() else "cpu")

        for name, func in cls.registry.items():
            if name in weights:
                raw_scores = func(generated, references, logits)
                normalized_scores = cls.normalize_score(name, raw_scores)
                total_rewards += weights[name] * normalized_scores

        return total_rewards

# Entries of the register and helper functions

@RewardRegistry.register("cider")
def cider_reward(generated: List[str], references: List[str], logits: torch.Tensor) -> torch.Tensor:
    """

    Returns a torch tensor containing the CIDEr score of each (generated caption, reference caption) pair.
    Args:
        generated (List[str)]): a batch of strings corresponding to the generated captions.
        references (List[str)]): a batch of reference captions, aligned with "generated".
        logits (torch.Tensor): the logits output by the LLM, in the format (batch * num_tokens * vocab_size).
    Return:
        A torch.Tensor of the CIDEr scores of each pair.
    """
    cider_scorer = Cider()
    image_ids = range(len(references))
    formatted_hypotheses = {image_id: [cap] for image_id, cap in zip(image_ids, generated)}
    formatted_references = {image_id: [cap] for image_id, cap in zip(image_ids, references)}
    _, scores = cider_scorer.compute_score(formatted_references, formatted_hypotheses)
    return torch.tensor(scores)

@RewardRegistry.register("length")
def length_penalization(generated: List[str], references: List[str], logits: torch.Tensor) -> torch.Tensor:
    """

    Returns a torch tensor containing the opposite of the length (# number of tokens) of each generated caption.
    This incentivizes the learned policy to output longer captions.
    Args:
        generated (List[str)]): a batch of strings corresponding to the generated captions.
        references (List[str)]): a batch of reference captions, aligned with "generated". (not used)
        logits (torch.Tensor): the logits output by the LLM, in the format (batch * num_tokens * vocab_size). (not used)
    Return:
        A torch.Tensor containing the opposite of the length of each generated caption.
    """
    return -torch.tensor([len(gen.encode('utf-8')) for gen in generated], dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")

def get_clip_embedding(word: str) -> np.ndarray:
    """

    Uses the preloaded CLIP model to obtain the embedding corresponding to the word "word".
    Args:
        word (str): a string, usually a word or a token extracted from a caption
    Return:
        The embedding corresponding to the word "word" in numpy Array format.
    """
    inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True, max_length=77)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    
    return outputs.cpu().numpy().flatten()

def calculate_clip_based_scores(gt_obj: List[List[str]], gen_obj: List[List[str]]) -> List[float]:
    """

    Using cosine similarity, computes a learning signal that incentivizes the policy to generate captions such that:
        - As many objects from the ground-truth caption are "aligned with" the corresponding generated caption objects;
        - The presence of objects in the generated caption that are not in the corresponding ground-truth caption is penalized; (hallucination)
        - The presence of objects in the ground-truth caption that are not in the corresponding generated caption is penalized; (oversights)
    Args:
        gt_obj (List[List[str]]): a list of lists of objects present in a batch of ground-truth captions.
        gen_obj (List[List[str]]): a list of lists of objects present in a batch of ground-truth captions.
    Return:
        scores (numpy Array): the learning signal associated to each generated caption.
    """
    scores = []

    for gt_list, gen_list in zip(gt_obj, gen_obj):
        gt_embeddings = [get_clip_embedding(word) for word in gt_list]
        gen_embeddings = [get_clip_embedding(word) for word in gen_list]

        # Reward for generated words close to ground truth words
        gen_to_gt_similarities = []
        for gen_emb in gen_embeddings:
            similarities = [1 - cosine(gen_emb, gt_emb) for gt_emb in gt_embeddings]
            gen_to_gt_similarities.append(max(similarities) if similarities else 0)
        
        coverage_score = sum(gen_to_gt_similarities) / len(gen_embeddings) if gen_embeddings else 0

        # Penalty for potential hallucinations (generated words far from all ground truth words)
        hallucination_penalty = sum(1 - sim for sim in gen_to_gt_similarities) / len(gen_embeddings) if gen_embeddings else 0

        # Penalty for oversights (ground truth words not close to any generated words)
        gt_to_gen_similarities = []
        for gt_emb in gt_embeddings:
            similarities = [1 - cosine(gt_emb, gen_emb) for gen_emb in gen_embeddings]
            gt_to_gen_similarities.append(max(similarities) if similarities else 0)
        
        oversight_penalty = sum(1 - sim for sim in gt_to_gen_similarities) / len(gt_embeddings) if gt_embeddings else 0

        # Combine scores
        final_score = coverage_score - 0.5 * (hallucination_penalty + oversight_penalty)
        scores.append(max(0, min(final_score, 1)))  # Ensure score is between 0 and 1

    return scores

@RewardRegistry.register("object_matching_clip")
def object_matching_reward(generated: List[str], references: List[str], logits: torch.Tensor) -> torch.Tensor:
    """

    Computes the final object matching score for each generated caption present in the batch.
    Args:
        generated (List[str)]): a batch of strings corresponding to the generated captions.
        references (List[str)]): a batch of reference captions, aligned with "generated".
        logits (torch.Tensor): the logits output by the LLM, in the format (batch * num_tokens * vocab_size). (not used)
    Return:
        torch.Tensor containing the score associated to each pair.
    """
    from scenegraph import scenegraphmaker
    gen_obj = scenegraphmaker(generated)
    gt_obj = scenegraphmaker(references)
    
    scores = calculate_clip_based_scores(gt_obj, gen_obj)
    return torch.tensor(scores, device=device)

@RewardRegistry.register("rouge")
def rouge_reward(generated: List[str], references: List[str], logits: torch.Tensor) -> torch.Tensor:
    """

    Computes the final object matching score for each generated caption present in the batch.
    Args:
        generated (List[str)]): a batch of strings corresponding to the generated captions.
        references (List[str)]): a batch of reference captions, aligned with "generated".
        logits (torch.Tensor): the logits output by the LLM, in the format (batch * num_tokens * vocab_size).
    Return:
        torch.Tensor containing the ROUGE score associated to each pair.
    """
    rouge_scorer = Rouge()
    scores = []
    for gen, ref in zip(generated, references):
        formatted_hypotheses = {0: [gen]}
        formatted_references = {0: [ref]}
        _, scores = rouge_scorer.compute_score(formatted_references, formatted_hypotheses)
    return torch.tensor(scores, device="cuda" if torch.cuda.is_available() else "cpu")

# InstructBLIP already includes a repetition penalty
"""@RewardRegistry.register("repetition_penalty")
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
    return torch.tensor(penalties, device="cuda" if torch.cuda.is_available() else "cpu")
"""

class ReadabilityReward:
    """
    A class for computing readability-based rewards using Flesch-Kincaid Grade Level
    and Gunning Fog Index.
    """
    def __init__(self) -> None:
        self.ih = 0

    def flesch_kincaid_grade(self, text: str) -> float:
        """
        Computes the Flesch-Kincaid Grade Level score for a given text.
        Args:
            text (str): The input text.
        Returns:
            float: The Flesch-Kincaid Grade Level score.
        """
        r = Readability(text)
        fk = r.flesch_kincaid()
        return fk.score

    def gunning_fog_index(self, text: str) -> float:
        """
        Computes the Gunning Fog Index score for a given text.
        Args:
            text (str): The input text.
        Returns:
            float: The Gunning Fog Index score.
        """
        r = Readability(text)
        gf = r.gunning_fog()
        return gf.score

    def compute_score(self, generated: List[str], references: List[str]) -> torch.Tensor:
        """
        Computes readability-based reward scores for generated text.
        Args:
            generated (List[str]): A batch of generated captions.
            references (List[str]): A batch of reference captions.
        Returns:
            torch.Tensor: A tensor containing readability reward scores.
        """
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

@RewardRegistry.register("grammar")
def grammar_reward(generated: List[str], references: List[str], logits: torch.Tensor) -> torch.Tensor:
    """
    Computes readability-based learning signals, to incentivize clear captions.
    Args:
        generated (List[str]): Batch of generated captions.
        references (List[str]): Batch of reference captions.
        logits (torch.Tensor): Logits output by the LLM.
    Returns:
        torch.Tensor: Grammar reward scores.
    """
    return ReadabilityReward().compute_score(generated, references)

@RewardRegistry.register("structural_complexity")
def structural_complexity_reward(generated: List[str], references: List[str], logits: torch.Tensor) -> torch.Tensor:
    """
    Computes structural complexity reward based on dependency depth and clause count, to incentivize complex captions in multiple clauses.
    Args:
        generated (List[str]): Batch of generated captions.
        references (List[str]): Batch of reference captions.
        logits (torch.Tensor): Logits output by the LLM.
    Returns:
        torch.Tensor: Structural complexity scores.
    """
    scores = []
    for caption in generated:
        doc = nlp(caption)
        depth = max(token.head.i - token.i for token in doc)
        num_clauses = len([sent for sent in doc.sents])
        score = depth + num_clauses
        scores.append(score)
    return torch.tensor(scores, device="cuda" if torch.cuda.is_available() else "cpu")

@RewardRegistry.register("meteor")
def meteor_reward(generated: List[str], references: List[str], logits: torch.Tensor) -> torch.Tensor:
    """
    Computes the METEOR reward score.
    Args:
        generated (List[str]): Batch of generated captions.
        references (List[str]): Batch of reference captions.
        logits (torch.Tensor): Logits output by the LLM.
    Returns:
        torch.Tensor: METEOR scores.
    """
    meteor_scorer = Meteor()
    scores = []
    for gen, ref in zip(generated, references):
        score, _ = meteor_scorer.compute_score({0: [ref]}, {0: [gen]})
        scores.append(score)
    return torch.tensor(scores, device="cuda" if torch.cuda.is_available() else "cpu")

@RewardRegistry.register("object_matching")
def object_matching_reward(generated: List[str], references: List[str], logits: torch.Tensor) -> torch.Tensor:
    """
    Computes a learning signal similar to object matching CLIP that relies on a simpler model than CLIP, and on "scene graphs".
    Args:
        generated (List[str]): Batch of generated captions.
        references (List[str]): Batch of reference captions.
        logits (torch.Tensor): Logits output by the LLM.
    Returns:
        torch.Tensor: Tensor of object matching scores.
    """
    gen_obj = scenegraphmaker(generated)
    gt_obj = scenegraphmaker(references)
    prop_scores, hallu_prop_scores = calculate_proportion(gt_obj, gen_obj)
    combined_scores = [p - h for p, h in zip(prop_scores, hallu_prop_scores)]
    return torch.tensor(combined_scores, device="cuda" if torch.cuda.is_available() else "cpu")

class RollingVocabKLDiv:
    def __init__(self, json_path: str, vocab_size: int = 30000, epsilon: float = 1e-7, update_freq: int = 100, smoothing_factor: float = 1.0):
        self.vocab_size = vocab_size
        self.epsilon = epsilon
        self.update_freq = update_freq
        self.smoothing_factor = smoothing_factor
        self.rolling_vocab_freq = Counter()
        self.word_pattern = re.compile(r'\b\w+\b')
        self.gt_vocab_prob = self._get_word_frequencies(self._load_json(json_path))
        self.gt_vocab_set = set(self.gt_vocab_prob.keys())
        self.kl_div_loss: float = 0.
        self.temp_captions = []
        self.total_words = 0

    def _load_json(self, file_path):
        with open(file_path) as f:
            data = json.load(f)
        return data['annotations']

    def _get_word_frequencies(self, annotations):
        text_data = ' '.join([annotation['caption'] for annotation in annotations]).lower()
        words = self.word_pattern.findall(text_data)
        word_frequencies = Counter(words)
        total_words = sum(word_frequencies.values())
        return {word: freq / total_words for word, freq in word_frequencies.items()}

    def _batch_update(self):
        new_words = self.word_pattern.findall(' '.join(self.temp_captions).lower())
        self.rolling_vocab_freq.update(new_words)
        self.total_words += len(new_words)
        if len(self.rolling_vocab_freq) > self.vocab_size:
            self.rolling_vocab_freq = Counter(dict(self.rolling_vocab_freq.most_common(self.vocab_size)))
        self.temp_captions = []

    def update_rolling_vocab_distribution(self, new_captions):
        self.temp_captions.extend(new_captions)
        if len(self.temp_captions) >= self.update_freq:
            self._batch_update()

    def compute_kl_div_loss(self):
        if self.temp_captions:
            self._batch_update()

        total_words = self.total_words + self.smoothing_factor * len(self.gt_vocab_prob)
        rolling_vocab_prob = {word: (freq + self.smoothing_factor) / total_words for word, freq in self.rolling_vocab_freq.items()}

        kl_div = sum(prob * np.log(prob / rolling_vocab_prob.get(word, self.epsilon))
                     for word, prob in self.gt_vocab_prob.items())

        self.kl_div_loss = kl_div
        return kl_div

# Initialize this once and reuse
kl_div_calculator = RollingVocabKLDiv('train2.json', vocab_size=30000)

@RewardRegistry.register("kl_divergence")
def kl_divergence_reward(generated: List[str], references: List[str], logits: torch.Tensor) -> torch.Tensor:
    """
    Computes KL divergence for the vocabulary distribution of the "vocab_size" latest tokens generated.
    Args:
        generated (List[str]): Batch of generated captions.
        references (List[str]): Batch of reference captions.
        logits (torch.Tensor): Logits output by the LLM.
    Returns:
        torch.Tensor: KL divergence loss values.
    """
    kl_div_calculator.update_rolling_vocab_distribution(generated)
    kl_div_loss = kl_div_calculator.compute_kl_div_loss()
    return torch.full((len(generated),), kl_div_loss, device="cuda" if torch.cuda.is_available() else "cpu")