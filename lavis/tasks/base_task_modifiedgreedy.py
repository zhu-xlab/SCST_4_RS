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
import deepspeed

to_pil = ToPILImage()

from graphbasedscore import CaptionEvaluator

evaluator = CaptionEvaluator()

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
            rl_start_epoch=0,
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
        rl_start_epoch=0,  # The epoch at which to start the RL refinement training
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

        t5_model.eval()

        ds_engine = deepspeed.init_inference(t5_model,
            mp_size=1,
            dtype=torch.bfloat16,
            replace_method="auto",
            replace_with_kernel_inject=True,
            enable_cuda_graph=True,
            quant={'enabled': True}
        )

        t5_model = ds_engine.module

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
                            param_group['lr'] = param_group['lr'] * 0.0005

                    model.train()

                    deepspeed_config = {
                        'train_batch_size': 128, 
                        'train_micro_batch_size_per_gpu': 128,
                        'steps_per_print': 100,
                        'gradient_accumulation_steps': 1,
                        """'optimizer': {
                            "type": "AdamW",
                            "params": {
                                "lr": 4e-5,
                                "betas": [0.9, 0.999],
                                "eps": 1e-8,
                                "weight_decay": 0.01
                            }
                        },"""
                        'bf16': {
                            'enabled': True,
                            'loss_scale': 0,
                            'loss_scale_window': 1000,
                            'hysteresis': 2,
                            'min_loss_scale': 1
                        },
                        'zero_optimization': {
                            'stage': 2,  # or 1 or 3 depending on your needs
                        },
                        'activation_checkpointing': {
                            'partition_activations': True,
                            'cpu_checkpointing': True,
                            'number_checkpoints': 2,
                            'synchronize_checkpoint_boundary': False,
                            'offload_optimizer': {
                                'device': 'cpu',
                                'pin_memory': True
                            }
                        }
                    }

                    # RL refinement training phase

                    output = model(samples)

                    # Compute log probabilities
                    logits = output['logits']
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                    """# Sample actions with temperature scaling
                    temperature = 0.5
                    noise_scale= 0.0
                    noise = torch.randn_like(logits) * noise_scale
                    scaled_logits = (logits + noise) / temperature
                    probs_2d = scaled_logits.softmax(dim=-1).view(-1, logits.size(-1))
                    action_2d = torch.multinomial(probs_2d, 1).squeeze(-1)

                    # Reshape actions back to original shape
                    action = action_2d.view(probs.size(0), probs.size(1))"""

                    def beam_search(logits, beam_size):
                        # Initialization
                        batch_size, seq_length, vocab_size = logits.size()
                        sequences = torch.zeros((batch_size, beam_size, seq_length), dtype=torch.long).cuda()  # for storing sequences
                        sequences_scores = torch.zeros((batch_size, beam_size)).cuda()  # for storing sequence scores
                        """sequences[:, :, 0] = torch.arange(beam_size).view(1, -1)  # initial sequences (can be changed)"""
                        start_token_index = 0  # Replace with the correct index for your model
                        sequences[:, :, 0] = start_token_index

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

                    # Assuming logits is already defined in your code
                    beam_size = 5
                    sequences, sequences_scores = beam_search(logits, beam_size)

                    # Get the best sequence (if you only want the best one)
                    best_sequences = sequences[:, 0, :]

                    # If you want the sequence (action) in the same shape as before
                    action = best_sequences

                    eos_token_id = 1  # Adjust based on your T5 model's tokenizer
                    max_length = 100  # Adjust based on your model and task requirements
                    terminal_states = torch.zeros(action.size(0), dtype=torch.bool).to(action.device)

                    for batch_idx, sequence in enumerate(action):
                        if eos_token_id in sequence or len(sequence) >= max_length:
                            terminal_states[batch_idx] = True

                    # Decode the sequences
                    generated_captions = t5_tokenizer.batch_decode(action, skip_special_tokens=True)

                    # Original greedy action
                    _, greedy_actions = logits.max(dim=-1)

                    caption_ids = action
                    generated_captions = t5_tokenizer.batch_decode(caption_ids, skip_special_tokens=True)
                    greedy_caption = t5_tokenizer.batch_decode(greedy_actions, skip_special_tokens=True)

                    #print('greedy 1st: ', greedy_caption[0:2])

                    # 3. Replace captions in samples['text_output'] using samples['image_id']
                    ground_truth_captions = []
                    for i_id in samples['image_id'].tolist():
                        ground_truth_captions.append(list(self.dict_ids[i_id]))
                    #print('ground_truth_captions: ', ground_truth_captions[0:2])
                    images_pil = [to_pil(img) for img in samples['image']]
                    dict_gen = evaluator.return_objects(generated_captions)
                    dict_gt, gt_obj = evaluator.return_objects_gt(ground_truth_captions)

                    #print('dict_gt: ', dict_gt)
                    #print('gt_obj: ', gt_obj)

                    # Convert each sublist into a single string
                    joined_texts = [' '.join(sublist) if sublist else '[EMPTY]' for sublist in gt_obj]

                    # Tokenize all strings at once
                    encoded = t5_tokenizer.batch_encode_plus(joined_texts, return_tensors='pt', padding=True, truncation=True)
                    input_ids = encoded['input_ids']

                    truncated_input_ids = []

                    # Iterate through each row of input_ids
                    for row in input_ids:
                        # Find the index of the first occurrence of 1
                        try:
                            idx = (row == 1).nonzero(as_tuple=True)[0].item()
                        except IndexError:  # If 1 is not found in the row
                            idx = len(row)
                        
                        # Keep everything before that index
                        truncated_row = row[:idx]
                        truncated_input_ids.append(truncated_row)

                    gt_obj_tokens = [ids for ids in truncated_input_ids]

                    epsilon = random.uniform(0.5, 1.0)

                    # Epsilon-greedy mask: True where exploration should be done, False for exploitation
                    epsilon_mask = torch.rand(logits.shape[:-1], device=logits.device) < epsilon

                    # Construct biased logits
                    large_negative = -1e9
                    biased_logits = large_negative * torch.ones_like(logits)

                    for batch_idx, tokens in enumerate(gt_obj_tokens):
                        for token in tokens:
                            biased_logits[batch_idx, :, token] = logits[batch_idx, :, token]

                    biased_probs = F.softmax(biased_logits, dim=-1)

                    # Sample from biased distribution
                    sampled_actions_from_biased = torch.multinomial(biased_probs.view(-1, biased_probs.size(-1)), 1).view(logits.shape[:-1])

                    # Combine the greedy actions and the epsilon-greedy samples
                    greedy_action = (torch.logical_not(epsilon_mask)) * greedy_actions + epsilon_mask * sampled_actions_from_biased
                    
                    """# 3. Replace captions in samples['text_output'] using samples['image_id']
                    ground_truth_captions = []
                    for i_id in samples['image_id'].tolist():
                        ground_truth_captions.append(list(self.dict_ids[i_id]))
                    images_pil = [to_pil(img) for img in samples['image']]
                    dict_gt, dict_gen, gt_obj = evaluator.return_objects(ground_truth_captions, generated_captions)

                    # Convert each sublist into a single string
                    joined_texts = [' '.join(sublist) if sublist else '[EMPTY]' for sublist in gt_obj]

                    # Tokenize all strings at once
                    encoded = t5_tokenizer.batch_encode_plus(joined_texts, return_tensors='pt', padding=True, truncation=True)
                    input_ids = encoded['input_ids']

                    truncated_input_ids = []

                    # Iterate through each row of input_ids
                    for row in input_ids:
                        # Find the index of the first occurrence of 1
                        try:
                            idx = (row == 1).nonzero(as_tuple=True)[0].item()
                        except IndexError:  # If 1 is not found in the row
                            idx = len(row)
                        
                        # Keep everything before that index
                        truncated_row = row[:idx]
                        truncated_input_ids.append(truncated_row)

                    # For each batch, set the generated caption to be the object tokens
                    caption_length = logits.size(1)  # Assuming logits shape is [batch_size, seq_length, vocab_size]
                    for batch_idx, object_tokens in enumerate(truncated_input_ids):
                        # Pad or truncate the object tokens to match the caption length
                        if len(object_tokens) < caption_length:
                            object_tokens = F.pad(object_tokens, (0, caption_length - len(object_tokens)))
                        else:
                            object_tokens = object_tokens[:caption_length]
                        
                        # Replace the generated caption
                        greedy_actions[batch_idx] = object_tokens

                    # Now, 'greedy_actions' contains the object tokens for each batch"""

                    # Compute CIDER score
                    from pycocoevalcap.cider.cider import Cider
                    cider_obj = Cider()

                    image_ids = range(len(generated_captions))
                    test_captions = {image_id: [cap] for image_id, cap in zip(image_ids, generated_captions)}
                    reference_captions = {image_id: [cap] for image_id, cap in zip(image_ids, samples["text_output"])}
                    greedy_captions = {image_id: [cap] for image_id, cap in zip(image_ids, greedy_caption)}

                    # Compute the CIDER score for sampled captions
                    _, cider_scores = cider_obj.compute_score(reference_captions, test_captions)

                    # Compute the CIDER score for greedy captions
                    _, greedy_cider_scores = cider_obj.compute_score(reference_captions, greedy_captions)

                    """image_ids = range(len(generated_captions))
                    test_captions = {image_id: [cap] for image_id, cap in zip(image_ids, generated_captions)}
                    reference_captions = {image_id: [cap] for image_id, cap in zip(image_ids, samples["text_output"])}
                    greedy_captions = {image_id: [cap] for image_id, cap in zip(image_ids, greedy_caption)}

                    # Compute the SPICE scores for test captions
                    _, cider_scores = evaluator_spice.spice_obj.compute_score(reference_captions, test_captions)

                    # Compute the SPICE scores for greedy captions
                    _, greedy_cider_scores = evaluator_spice.spice_obj.compute_score(reference_captions, greedy_captions)

                    cider_scores = [s['All']['f'] for s in cider_scores]
                    greedy_cider_scores = [s['All']['f'] for s in greedy_cider_scores]"""

                    greedy_caption = t5_tokenizer.batch_decode(greedy_action, skip_special_tokens=True)

                    dict_greedy = evaluator.return_objects(greedy_caption)
                    dict_gt_copy = copy.deepcopy(dict_gt)
                    custom_scores = evaluator.compute_score(images_pil, dict_gt, dict_gen)
                    greedy_custom_scores = evaluator.compute_score(images_pil, dict_gt_copy, dict_greedy)

                    print('gen: ', generated_captions[0:2])
                    print('greedy: ', greedy_caption[0:2])

                    # CIDEr and custom scores
                    """beta = 0.1
                    average_errors = beta * np.array(custom_scores) + (1 - beta) * np.array(cider_scores)
                    greedy_errors = beta * np.array(greedy_custom_scores) + (1 - beta) * np.array(greedy_cider_scores)"""

                    # Custom scores and length

                    beta = 0.1
                    length_penalty = np.array([len(cap) if term else 0 for cap, term in zip(generated_captions, terminal_states)])
                    average_errors = np.array(custom_scores) - np.array(cider_scores) - beta * length_penalty

                    greedy_length_penalty = np.array([len(cap) if term else 0 for cap, term in zip(greedy_caption, terminal_states)])
                    greedy_errors = np.array(greedy_custom_scores) - np.array(greedy_cider_scores) - beta * greedy_length_penalty

                    rewards = torch.tensor(average_errors).to(logits.device) - torch.tensor(greedy_errors).to(logits.device)
                    #alpha = 0.01
                    alpha = 0.01
                    running_mean_reward = (1 - alpha) * running_mean_reward + alpha * rewards.mean().item()
                    advantage = (rewards - running_mean_reward).detach()
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                    # Log advantage statistics
                    #print(f"Advantage - Avg: {advantage.mean().item()}, Min: {advantage.min().item()}, Max: {advantage.max().item()}")

                    # Entropy regularization
                    entropy = -(probs * log_probs).sum(-1).mean()
                    entropy_weight = 0.01

                    # Calculate the loss
                    """unif_like_probs = torch.rand(probs.shape).cuda()
                    eps_surrogate = 0.98
                    surrogate_distribution = probs * eps_surrogate + unif_like_probs * (1 - eps_surrogate)"""
                    ###SUM instead of MEAN
                    #policy_loss = -(probs.gather(-1, action.unsqueeze(-1)) / (surrogate_distribution.gather(-1, action.unsqueeze(-1))) * advantage * log_probs.gather(-1, action.unsqueeze(-1))).mean()
                    policy_loss = -(advantage * log_probs.gather(-1, action.unsqueeze(-1))).mean()
                    #importance_weights = probs.gather(-1, action.unsqueeze(-1)) / (surrogate_distribution.gather(-1, action.unsqueeze(-1)))
                    #print(f"Importance weights - Avg: {importance_weights.mean().item()}, Min: {importance_weights.min().item()}, Max: {importance_weights.max().item()}")
                    loss = policy_loss - entropy_weight * entropy

                    """print("policy_loss ", policy_loss)
                    print("loss ", loss)"""

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
