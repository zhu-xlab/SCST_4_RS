"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample
from lavis.datasets.datasets.dataloader_utils import IterLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import T5TokenizerFast
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration

class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

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
        rl_start_epoch=2,
    ):

        data_loader_new = DataLoader(
            data_loader._dataloader.dataset, 
            batch_size=data_loader._dataloader.batch_size // 2,  
            num_workers=data_loader._dataloader.num_workers,
            collate_fn=data_loader._dataloader.collate_fn,
            drop_last=data_loader._dataloader.drop_last,
            timeout=data_loader._dataloader.timeout,
            sampler=data_loader._dataloader.sampler,
            pin_memory=data_loader._dataloader.pin_memory
            )
        if epoch >= rl_start_epoch:
            return self._train_inner_loop(
                epoch=epoch,
                iters_per_epoch=len(data_loader_new),
                model=model,
                data_loader=data_loader,
                optimizer=optimizer,
                scaler=scaler,
                lr_scheduler=lr_scheduler,
                log_freq=log_freq,
                cuda_enabled=cuda_enabled,
                accum_grad_iters=accum_grad_iters,
                rl_start_epoch=2,
            )
        else:
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
                rl_start_epoch=2,
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
        rl_start_epoch=2,  # The epoch at which to start the RL refinement training
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

        if epoch >= rl_start_epoch:
            data_loader_new = DataLoader(
            data_loader._dataloader.dataset, 
            batch_size=data_loader._dataloader.batch_size // 2,  
            num_workers=data_loader._dataloader.num_workers,
            collate_fn=data_loader._dataloader.collate_fn,
            drop_last=data_loader._dataloader.drop_last,
            timeout=data_loader._dataloader.timeout,
            sampler=data_loader._dataloader.sampler,
            pin_memory=data_loader._dataloader.pin_memory
            )
            data_loader_new = IterLoader(data_loader_new)
            data_loader_new = iter(data_loader_new)

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

            if epoch < rl_start_epoch or data_loader_new is None:
                samples = next(data_loader)
            else:
                samples = next(data_loader_new)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            import torch.nn as nn
            import torch.nn.functional as F

            class Discriminator(nn.Module):
                def __init__(self, embed_size, hidden_size, vocab_size):
                    super(Discriminator, self).__init__()

                    self.embedding = nn.Embedding(vocab_size, embed_size)
                    
                    # LSTM layer
                    self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

                    # Fully connected layers
                    self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
                    self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
                    self.fc3 = nn.Linear(hidden_size // 4, 1)

                    # Dropout for regularization
                    self.dropout = nn.Dropout(0.3)

                def forward(self, captions):
                    embedded = self.embedding(captions)
                    _, (hidden, _) = self.lstm(embedded)
                    
                    # Pass LSTM's final hidden state through fully connected layers
                    x = self.dropout(F.relu(self.fc1(hidden.squeeze(0))))
                    x = self.dropout(F.relu(self.fc2(x)))
                    x = self.fc3(x)

                    return torch.sigmoid(x)

            # Instantiate the discriminator and its optimizer
            embed_size = 256  # Set to desired value
            hidden_size = 512  # Set to desired value
            vocab_size = 32128  # Set to desired value

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            discriminator = Discriminator(embed_size, hidden_size, vocab_size).to(device)
            optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=3e-3)

            # Helper function for curriculum learning
            def filter_samples_by_length(samples, max_length):
                if isinstance(samples["text_output"], list) and isinstance(samples["text_output"][0], str):
                    # Convert raw string captions to tokenized sequences
                    text_output_tensor = torch.tensor(t5_tokenizer(samples["text_output"], return_tensors="pt", truncation=True, padding=True, max_length=max_caption_length)["input_ids"])
                else:
                    text_output_tensor = torch.tensor(samples["text_output"]) if isinstance(samples["text_output"], list) else samples["text_output"]

                # Convert the boolean tensor into integers (0 or 1) and then sum
                mask = torch.sum((text_output_tensor.ne(0)).int(), dim=1) <= max_length

                # Filtering different types of items in the samples dictionary
                filtered_samples = {}
                for k, v in samples.items():
                    if torch.is_tensor(v):  # if tensor, apply the mask directly
                        filtered_samples[k] = v[mask]
                    elif isinstance(v, list):  # if list, use list comprehension
                        filtered_samples[k] = [v[i] for i, m in enumerate(mask) if m]
                    else:  # if neither tensor nor list, simply copy the value
                        filtered_samples[k] = v

                return filtered_samples

            initial_max_length = 10  # Initialize as per your data
            length_increase_interval = 1  # Change as needed
            max_caption_length = initial_max_length

            #use_amp = False

            with torch.cuda.amp.autocast(enabled=use_amp):
                if epoch < rl_start_epoch:
                    # Initial training phase
                    loss, loss_dict = self.train_step(model=model, samples=samples)
                    loss /= accum_grad_iters
                else:
                    if epoch >= rl_start_epoch:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = param_group['lr'] * 0.001

                    # Curriculum Learning
                    max_caption_length = min(initial_max_length + epoch // length_increase_interval, max_caption_length)
                    samples = filter_samples_by_length(samples, max_caption_length)

                    model.train()
                    output = model(samples)

                    # Compute log probabilities
                    logits = output['logits']
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                    # Sample K captions with temperature scaling
                    K = 100  # or any other desired value
                    temperature = 0.7
                    scaled_logits = logits / temperature
                    probs_2d = scaled_logits.softmax(dim=-1).view(-1, logits.size(-1))
                    actions_2d = torch.multinomial(probs_2d, K)

                    # Reshape actions back to original shape
                    actions = actions_2d.view(probs.size(0), probs.size(1), K)

                    # Compute CIDER score
                    from pycocoevalcap.cider.cider import Cider
                    cider_obj = Cider()

                    # Assuming you have a list of image IDs for each caption
                    image_ids = range(len(actions)) 

                    # Prepare the inputs for the CIDER scorer
                    reference_captions = {image_id: [cap] for image_id, cap in zip(image_ids, samples["text_output"])}

                    cider_scores_list = []
                    for k in range(K):
                        caption_ids = actions[:, :, k]
                        caption = t5_tokenizer.batch_decode(caption_ids, skip_special_tokens=True)
                        test_captions = {image_id: [cap] for image_id, cap in zip(image_ids, caption)}
                        _, cider_scores_k = cider_obj.compute_score(reference_captions, test_captions)
                        cider_scores_list.append(cider_scores_k)

                    cider_scores_array = np.array(cider_scores_list)
                    cider_scores_tensor = torch.tensor(cider_scores_array).to(logits.device)
                    baselines = (cider_scores_tensor.sum(0) - cider_scores_tensor) / (K - 1)

                    rewards = cider_scores_tensor - baselines
                    advantage = (rewards - rewards.mean(dim=0)).detach()
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    
                    entropy = -(probs * log_probs).sum(-1).mean()
                    entropy_weight = 0.001

                    expanded_log_probs = log_probs.unsqueeze(-2).expand(-1, -1, K, -1)
                    policy_loss = -(advantage * expanded_log_probs.gather(-1, actions.unsqueeze(-1))).mean()
                    loss = policy_loss - entropy_weight * entropy

                    # Tokenize the raw string captions first
                    tokenized_real_captions = t5_tokenizer(samples["text_output"], return_tensors="pt", truncation=True, padding=True, max_length=max_caption_length)["input_ids"]
                    real_captions = tokenized_real_captions.to(device)
                    labels_real = torch.ones(real_captions.size(0), 1).to(device)
                    
                    # Since `actions` is already a tensor, no need to convert `fake_captions` again.
                    random_indices = torch.randint(0, K, (actions.size(0),)).to(device)  # Here's the fix.
                    fake_captions = actions[torch.arange(actions.size(0)), :, random_indices].to(device)
                    labels_fake = torch.zeros(fake_captions.size(0), 1).to(device)

                    logits_real = discriminator(real_captions)
                    logits_fake = discriminator(fake_captions)

                    loss_real = F.binary_cross_entropy_with_logits(logits_real, labels_real)
                    loss_fake = F.binary_cross_entropy_with_logits(logits_fake, labels_fake)

                    loss_discriminator = loss_real + loss_fake
                    optimizer_discriminator.zero_grad()
                    loss_discriminator.backward()
                    optimizer_discriminator.step()

                    logits_fake_for_generator = discriminator(fake_captions)
                    loss_generator = F.binary_cross_entropy_with_logits(logits_fake_for_generator, labels_real)
                    loss += loss_generator

                    max_norm = 5.0
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
