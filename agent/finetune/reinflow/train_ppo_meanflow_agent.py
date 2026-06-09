# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
PPO fine-tuning for MeanFlow policies with low-dimensional observations (gym).
"""
import os
import logging
log = logging.getLogger(__name__)
from tqdm import tqdm as tqdm
import numpy as np
import torch
from agent.finetune.reinflow.train_ppo_shortcut_agent import TrainPPOShortCutAgent
from model.flow.ft_ppo.ppomeanflow import PPOMeanFlow
from agent.finetune.reinflow.buffer import PPOFlowBuffer


class TrainPPOMeanFlowAgent(TrainPPOShortCutAgent):
    """
    Training agent for PPO fine-tuning of MeanFlow policies with low-dimensional observations.

    Inherits from TrainPPOShortCutAgent and overrides MeanFlow-specific components.
    The main differences are in the policy network and sampling procedure.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.buffer_device = 'cpu'
        self.grad_accumulate = cfg.train.get("grad_accumulate", 1)
        log.info("Initialized MeanFlow PPO training agent with low-dim observations")

        # MeanFlow uses 5 inference steps by default (as per original paper)
        # This can be overridden in config
        self.inference_steps = cfg.model.inference_steps if hasattr(cfg.model, 'inference_steps') else 5

        # Adjust some parameters for MeanFlow stability
        self.initial_ratio_error_threshold = 1e-5  # MeanFlow can be more stable

    def agent_update(self, verbose=True):
        """
        Agent update specifically for MeanFlow PPO.

        The main difference from shortcut flow is the underlying sampling and
        log probability computation, but the PPO loss computation remains the same.
        """
        clipfracs_list = []
        noise_std_list = []
        actor_norm = 0.0
        critic_norm = 0.0

        for update_epoch, batch_id, minibatch in self.minibatch_generator() if not self.repeat_samples else self.minibatch_generator_repeat():
            # Minibatch gradient descent for MeanFlow
            self.model: PPOMeanFlow

            pg_loss, entropy_loss, v_loss, bc_loss, \
            clipfrac, approx_kl, ratio, \
            oldlogprob_min, oldlogprob_max, oldlogprob_std, \
                newlogprob_min, newlogprob_max, newlogprob_std, \
                noise_std, newQ_values = self.model.loss(*minibatch,
                                                    use_bc_loss=self.use_bc_loss,
                                                    bc_loss_type=self.bc_loss_type,
                                                    normalize_denoising_horizon=self.normalize_denoising_horizon,
                                                    normalize_act_space_dimension=self.normalize_act_space_dim,
                                                    verbose=verbose,
                                                    clip_intermediate_actions=self.clip_intermediate_actions,
                                                    account_for_initial_stochasticity=self.account_for_initial_stochasticity)
            self.approx_kl = approx_kl

            if verbose:
                log.info(f"MeanFlow update_epoch={update_epoch}/{self.update_epochs}, batch_id={batch_id}/{max(1, self.total_steps // self.batch_size)}, ratio={ratio:.3f}, clipfrac={clipfrac:.3f}, approx_kl={self.approx_kl:.2e}")

            # Check for potential bugs in the first update
            if update_epoch == 0 and batch_id == 0 and np.abs(ratio - 1.00) > self.initial_ratio_error_threshold:
                log.warning(f"Warning: MeanFlow ratio={ratio} not 1.00 when update_epoch==0 and batch_id==0, there might be bugs in the implementation!")

            # Adaptive learning rate based on KL divergence
            if self.target_kl and self.lr_schedule == 'adaptive_kl':
                self.update_lr_adaptive_kl(self.approx_kl)

            # Compute total loss
            loss = pg_loss + entropy_loss * self.ent_coef + v_loss * self.vf_coef + bc_loss * self.bc_coeff

            clipfracs_list += [clipfrac]
            noise_std_list += [noise_std]

            loss.backward()

            # Gradient accumulation support
            if (batch_id + 1) % self.grad_accumulate == 0:
                # Compute gradient norms for monitoring
                actor_norm = torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), max_norm=float('inf'))
                critic_norm = torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), max_norm=float('inf'))
                if verbose:
                    log.info(f"MeanFlow before clipping: actor_norm={actor_norm:.2e}, critic_norm={critic_norm:.2e}")

                # Update actor after critic warmup
                if self.itr >= self.n_critic_warmup_itr:
                    if self.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()

                # Update critic
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # Clear gradients
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                if verbose:
                    log.info(f"MeanFlow grad update at batch {batch_id}")
                    log.info(f"MeanFlow approx_kl: {approx_kl}, update_epoch: {update_epoch}/{self.update_epochs}, num_batch: {self.total_steps // self.batch_size}")

        # Aggregate statistics
        clip_fracs = np.mean(clipfracs_list)
        noise_stds = np.mean(noise_std_list)

        # Training metrics dictionary
        self.train_ret_dict = {
            "loss": loss,
            "pg loss": pg_loss,
            "value loss": v_loss,
            "entropy_loss": entropy_loss,
            "bc_loss": bc_loss,
            "approx kl": self.approx_kl,
            "ratio": ratio,
            "clipfrac": clip_fracs,
            "explained variance": self.explained_var,
            "old_logprob_min": oldlogprob_min,
            "old_logprob_max": oldlogprob_max,
            "old_logprob_std": oldlogprob_std,
            "new_logprob_min": newlogprob_min,
            "new_logprob_max": newlogprob_max,
            "new_logprob_std": newlogprob_std,
            "actor_norm": actor_norm,
            "critic_norm": critic_norm,
            "actor lr": self.actor_optimizer.param_groups[0]["lr"],
            "critic lr": self.critic_optimizer.param_groups[0]["lr"],
            "min_logprob_noise_std": self.model.min_logprob_denoising_std,
            "min_sampling_noise_std": self.model.min_sampling_denoising_std,
            "noise_std": noise_stds,
            "Q_values": newQ_values
        }

    # run() inherited from TrainPPOShortCutAgent - uses get_samples_logprobs() correctly
