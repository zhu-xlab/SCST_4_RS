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

import torch
import torch.nn as nn
import torch.distributed as dist
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample

import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from transformers import T5TokenizerFast
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from tqdm import tqdm
import numpy as np
import time

to_pil = ToPILImage()

from graphbasedscore import CaptionEvaluator

evaluator = CaptionEvaluator()

class ValueNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, sequence):
        embedded_seq = self.embedding(sequence)
        _, hidden = self.rnn(embedded_seq)
        value = self.fc(hidden.squeeze(0))
        return value.squeeze(-1)

def compute_gae(next_value, rewards, masks, values, gamma=0.99, lambda_gae=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
    - next_value (Tensor): Value estimate of the next state.
    - rewards (Tensor): Reward received after taking the action.
    - masks (Tensor): Masks to indicate whether it's a terminal state.
    - values (Tensor): Value estimates for the current states.
    - gamma (float): Discount factor.
    - lambda_gae (float): GAE parameter.

    Returns:
    - returns (Tensor): The computed returns.
    - advantages (Tensor): The computed advantages.
    """
    gae = 0
    returns = []
    advantages = []

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value[step] * masks[step] - values[step]
        gae = delta + gamma * lambda_gae * masks[step] * gae
        returns.insert(0, gae + values[step])
        advantages.insert(0, gae)

    returns = torch.cat(returns)
    advantages = torch.cat(advantages)
    
    # Normalize the advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages

class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"
        with open("/mnt/SSD2/thomas/LAVIS/train2.json", "r") as file:
            coco_data = json.load(file)

        # 2. Build a dictionary mapping from image_id to captions.
        image_id_to_captions = {}
        for annotation in coco_data["annotations"]:
            image_id = annotation["image_id"]
            caption = annotation["caption"]
                                
            if image_id not in image_id_to_captions:
                image_id_to_captions[image_id] = set()  # using a set to ensure non-redundancy
                                
            image_id_to_captions[image_id].add(caption)
        self.dict_ids = image_id_to_captions

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

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
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
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
        accum_grad_iters=1,
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
            rl_start_epoch=1,
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
        accum_grad_iters=0,
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
        accum_grad_iters=1,
        rl_start_epoch=1,  # The epoch at which to start the RL refinement training
    ):

        running_mean_reward = 0

        use_amp = scaler is not None

        t5_model = 'google/flan-t5-xl'

        t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model, truncation_side='right')

        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )

        if not hasattr(data_loader, "__next__"):
            data_loader = iter(data_loader)

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

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            use_amp = False

            with torch.cuda.amp.autocast(enabled=use_amp):
                if epoch < rl_start_epoch:
                    # Initial training phase
                    loss, loss_dict = self.train_step(model=model, samples=samples)
                    loss /= accum_grad_iters
                else:
                    if epoch >= rl_start_epoch:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = param_group['lr'] * 0.001

                    model.train()

                    # RL refinement training phase

                    # Compute log probabilities
                    #logits = output['logits']
                    vocab_size, embedding_dim, hidden_dim = logits.shape

                    value_network = ValueNetwork(vocab_size, embedding_dim, hidden_dim)
                    optimizer_value = torch.optim.Adam(value_network.parameters(), lr=1e-4)

                    # PPO parameters
                    ppo_epochs = 4
                    ppo_clip = 0.2
                    gamma = 0.99
                    lambda_gae = 0.95

                    num_iterations = 4

                    # Data collection
                    states, actions, log_probs_old, rewards, masks = [], [], [], [], []

                    for _ in range(num_iterations):
                        output = model(samples)
                        logits = output['logits']

                        # Sample actions using your existing method (e.g., beam search, greedy, etc.)
                        action = beam_search(logits, beam_size)

                        # Store states, actions, and log probabilities
                        states.append(samples)
                        actions.append(action)
                        log_probs_old.append(torch.nn.functional.log_softmax(logits, dim=-1).gather(-1, action.unsqueeze(-1)))

                        # Decode the sequences
                        generated_captions = t5_tokenizer.batch_decode(action, skip_special_tokens=True)

                        # Original greedy action
                        _, greedy_actions = logits.max(dim=-1)
                        greedy_caption = t5_tokenizer.batch_decode(greedy_actions, skip_special_tokens=True)

                        # Compute rewards and masks
                        ground_truth_captions = [list(self.dict_ids[i_id]) for i_id in samples['image_id'].tolist()]
                        dict_gen = evaluator.return_objects(generated_captions)
                        dict_gt, gt_obj = evaluator.return_objects_gt(ground_truth_captions)

                        # CIDEr score computation
                        cider_obj = Cider()
                        image_ids = range(len(generated_captions))
                        test_captions = {image_id: [cap] for image_id, cap in zip(image_ids, generated_captions)}
                        reference_captions = {image_id: [cap] for image_id, cap in zip(image_ids, samples["text_output"])}
                        greedy_captions = {image_id: [cap] for image_id, cap in zip(image_ids, greedy_caption)}

                        _, cider_scores = cider_obj.compute_score(reference_captions, test_captions)
                        _, greedy_cider_scores = cider_obj.compute_score(reference_captions, greedy_captions)

                        # Compute custom scores
                        custom_scores = evaluator.compute_score(images_pil, dict_gt, dict_gen)
                        greedy_custom_scores = evaluator.compute_score(images_pil, dict_gt_copy, dict_greedy)

                        # Combining the scores
                        beta = 0.3
                        average_errors = beta * np.array(custom_scores) + (1 - beta) * np.array(cider_scores)
                        greedy_errors = beta * np.array(greedy_custom_scores) + (1 - beta) * np.array(greedy_cider_scores)

                        # Calculate final reward for PPO update
                        rewards_batch = torch.tensor(average_errors).to(logits.device) - torch.tensor(greedy_errors).to(logits.device)
                        rewards.append(rewards_batch)

                        # Masks for terminal states (assuming all non-terminal for now)
                        masks_batch = torch.ones_like(rewards_batch)
                        masks.append(masks_batch)

                    """probs = torch.nn.functional.softmax(logits, dim=-1)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)"""

                    def beam_search(logits, beam_size):
                        # Initialization
                        batch_size, seq_length, vocab_size = logits.size()
                        sequences = torch.zeros((batch_size, beam_size, seq_length), dtype=torch.long).cuda()  # for storing sequences
                        sequences_scores = torch.zeros((batch_size, beam_size)).cuda()  # for storing sequence scores
                        sequences[:, :, 0] = torch.arange(beam_size).view(1, -1)  # initial sequences (can be changed)

                        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cuda()
                        
                        for t in range(1, seq_length):
                            # For each timestep, compute scores for each possible next token
                            scores = log_probs[:, t, :].unsqueeze(1) + sequences_scores.unsqueeze(-1)
                            scores = scores.view(batch_size, -1)  # flatten out the beam dimension
                            
                            # Select top-k scores
                            top_scores, top_indices = scores.topk(beam_size, dim=1)
                            
                            # Update sequences and sequences_scores with top-k sequences
                            sequences_scores = top_scores
                            prev_seq_indices = top_indices // vocab_size  # Which sequence it came from
                            next_word_indices = top_indices % vocab_size  # Which word was added
                            
                            # Extract the sequences up to (but not including) the current time step.
                            prev_sequences = torch.zeros_like(sequences)
                            for i in range(batch_size):
                                for j in range(beam_size):
                                    prev_sequences[i, j] = sequences[i, prev_seq_indices[i, j]]

                            # Now, ensure that next_word_indices have the shape (batch_size, beam_size).
                            prev_sequences[:, :, t] = next_word_indices

                            # Assign the updated sequences back to the original sequences tensor.
                            sequences = prev_sequences
                            
                        return sequences, sequences_scores

                    states = torch.cat(states)
                    actions = torch.cat(actions)
                    log_probs_old = torch.cat(log_probs_old)
                    rewards = torch.tensor(rewards)
                    masks = torch.tensor(masks)

                    def extract_best_sequence(sequences, sequences_scores, top_k=1):
                        """
                        Extracts the best sequence(s) based on the scores.

                        Args:
                            sequences (Tensor): The sequences generated by beam search.
                            sequences_scores (Tensor): The scores for each sequence.
                            top_k (int): Number of top sequences to return.

                        Returns:
                            Tensor: The best sequence(s).
                        """
                        _, top_indices = sequences_scores.topk(top_k, dim=1)
                        best_sequences = sequences.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, sequences.size(-1)))
                        return best_sequences.squeeze(1)  # Remove the beam dimension if top_k is 1

                    # Assuming you have your logits and beam_size
                    sequences, sequences_scores = beam_search(logits, beam_size)

                    # Extract the best sequence to represent the next state
                    next_state = extract_best_sequence(sequences, sequences_scores, top_k=1)

                    # Calculate returns and advantages
                    values = value_network(states)
                    next_value = value_network(next_state)
                    returns, advantages = compute_gae(next_value, rewards, masks, values, gamma, lambda_gae)

                    for _ in range(ppo_epochs):
                        # Recalculate log_probs for actions under the current policy
                        current_output = model(states)
                        current_logits = current_output['logits']
                        current_log_probs = torch.nn.functional.log_softmax(current_logits, dim=-1).gather(-1, actions.unsqueeze(-1))

                        # Calculate the PPO objective
                        ratios = torch.exp(current_log_probs - log_probs_old)
                        surr1 = ratios * advantages
                        surr2 = torch.clamp(ratios, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantages
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # Recompute values for the current states
                        values = value_network(states)
                        value_loss = F.mse_loss(returns, values)

                        # Total loss
                        loss = policy_loss + value_loss

                        # Perform optimization
                        optimizer_value.zero_grad()
                        loss.backward()
                        optimizer_value.step()

                    max_norm = 5.0

                    # Gradient clipping (after computing gradients)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                    loss /= accum_grad_iters

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }


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
