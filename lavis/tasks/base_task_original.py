"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
import os
import copy
import random
from typing import Dict, List, Any, Tuple

import torch
import torch.distributed as dist
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample

import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torch.utils.tensorboard import SummaryWriter
from transformers import T5TokenizerFast
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from tqdm import tqdm
import numpy as np
import time
import deepspeed
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from collections import Counter

# Import from our custom reward and evolutionary files
from rewards import RewardRegistry
from evolutionary_algorithm import EvolutionaryAlgorithm

to_pil = ToPILImage()

from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider

from graphbasedscore import CaptionEvaluator

evaluator = CaptionEvaluator()

writer = SummaryWriter('runs/test_LR_plateau2_3epochs')

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

def calculate_proportion(gt_obj, gen_obj):
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

class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"
        self.old_action_probs = None
        self.spice_obj = Meteor()
        with open("/mnt/SSD2/thomas/LAVIS/train2.json", "r") as file:
            coco_data = json.load(file)

        image_id_to_captions = {}
        for annotation in coco_data["annotations"]:
            image_id = annotation["image_id"]
            caption = annotation["caption"]
                                
            if image_id not in image_id_to_captions:
                image_id_to_captions[image_id] = set()
                                
            image_id_to_captions[image_id].add(caption)
        self.dict_ids = image_id_to_captions

        # Initialize EvolutionaryAlgorithm
        reward_component_names = list(RewardRegistry.registry.keys())
        self.evolutionary_algorithm = EvolutionaryAlgorithm(num_individuals=20, reward_component_names=reward_component_names)

        # Initialize tokenizer
        self.tokenizer = T5TokenizerFast.from_pretrained('google/flan-t5-xl', truncation_side='right')

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        datasets = dict()
        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]
            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()
            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        output = model(samples)
        loss_dict = {}
        for k,v in output.items():
            if "loss" in k:
                loss_dict[k] = v
        return output["loss"], loss_dict

    def valid_step(self, model, samples):
        results = []
        captions = model.generate(samples)
        
        for caption, sample in zip(captions, samples):
            results.append({
                "caption": caption,
                "image_id": sample["image_id"],
            })
        
        return results

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=2,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=len(data_loader),
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
            rl_start_epoch=0,
            global_step=0,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=2,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=2,
        rl_start_epoch=2,
        global_step=0,
    ):
        model.train()
        
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            inner_epoch = epoch
        else:
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            if i >= iters_per_epoch:
                break

            samples = next(iter(data_loader))
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                if epoch < rl_start_epoch:
                    # Initial training phase
                    loss, loss_dict = self.train_step(model=model, samples=samples)
                    loss /= accum_grad_iters
                else:
                    # RL refinement training phase
                    model.train()
                    output = model(samples)
                    logits = output['logits']

                    # Sample actions
                    action = self.sample_actions(logits)
                    greedy_actions = logits.argmax(dim=-1)

                    # Decode captions
                    generated_captions = self.decode_captions(action)
                    greedy_captions = self.decode_captions(greedy_actions)
                    ground_truth_captions = samples['text_output']

                    # Compute rewards using RewardRegistry
                    rewards = RewardRegistry.compute_total_reward(generated_captions, ground_truth_captions, self.evolutionary_algorithm.ema_weights, logits)
                    baseline_rewards = RewardRegistry.compute_total_reward(greedy_captions, ground_truth_captions, self.evolutionary_algorithm.ema_weights, logits)

                    advantage = rewards - baseline_rewards

                    # Calculate loss
                    log_probs = F.log_softmax(logits, dim=-1)
                    policy_loss = -(advantage.unsqueeze(-1) * log_probs.gather(-1, action.unsqueeze(-1))).mean()

                    # Add entropy regularization
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * log_probs).sum(-1).mean()
                    entropy_weight = 0.01

                    # Add KL divergence
                    kl_div_calculator = RollingVocabKLDiv('train2.json', vocab_size=10000)
                    kl_weight = 1000.
                    kl_div_calculator.update_rolling_vocab_distribution(generated_captions)
                    kl_div_loss = kl_div_calculator.compute_kl_div_loss()

                    loss = policy_loss - entropy_weight * entropy + kl_weight * kl_div_loss
                    loss /= accum_grad_iters

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % accum_grad_iters == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:    
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            # Optimize reward weights periodically
            if i % 100 == 0:
                self.evolutionary_algorithm.evolve(model, data_loader)

            global_step += 1

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    def sample_actions(self, logits):
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.size(0), -1)

    def decode_captions(self, caption_ids):
        return self.tokenizer.batch_decode(caption_ids, skip_special_tokens=True)

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file

def load_config():
    import yaml
    from lavis.common.config import Config
    
    config_path = "path/to/your/config.yaml"  # Replace with actual path
    with open(config_path, 'r') as f:
        config_yaml = yaml.safe_load(f)
    
    config = Config(config_yaml)
    return config

def main(cfg):
    task = BaseTask.setup_task()
    model = task.build_model(cfg)
    datasets = task.build_datasets(cfg)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.run_cfg.init_lr)
    
    # Setup learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.run_cfg.max_epoch
    )
    
    # Setup gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(cfg.run_cfg.max_epoch):
        task.train_epoch(
            epoch=epoch,
            model=model,
            data_loader=datasets['train'],
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            cuda_enabled=True,
            log_freq=cfg.run_cfg.log_freq,
            accum_grad_iters=cfg.run_cfg.accum_grad_iters,
        )
        
        # Evaluation
        eval_results = task.evaluation(
            model=model,
            data_loader=datasets['val'],
            cuda_enabled=True
        )
        
        # Save checkpoint
        if is_main_process():
            task.save_result(
                result=eval_results,
                result_dir=cfg.output_dir,
                filename=f"eval_epoch_{epoch}",
                remove_duplicate="image_id"
            )
            
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "config": cfg,
            }
            torch.save(
                checkpoint,
                os.path.join(cfg.output_dir, f"checkpoint_{epoch}.pth")
            )
    
    logging.info("Training complete.")

if __name__ == "__main__":
    config = load_config()
    main(config)