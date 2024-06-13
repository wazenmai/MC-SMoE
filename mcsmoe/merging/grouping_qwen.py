import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from types import MethodType

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Qwen2MoeForCausalLM, Qwen2MoeConfig
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock, Qwen2MoeMLP

from .utils import generate_random_group_labels
from mcsmoe.utils.constants import FP32_EPS
from mcsmoe.models.qwen import merged_qwen2moe_forward

SIMILARITY_MAPPING_FUNCTION = {
    "cosine": lambda x, y: (F.cosine_similarity(x, y, dim=-1, eps=FP32_EPS) + 1).item() / 2,
    "mse": lambda x, y: 1 / (1 + 0.1 * torch.log(F.mse_loss(x, y, reduction="sum"))).item(),
}

LEGAL_SIMILARITY_BASES = ["weight", "feature", "feature.abs", "weight-feature", "gradient", "weight-gradient",
                          "router-logits", "router-weight", "router-weight-feature", "mse", "random",
                          "feature-correlation.lsa", "feature-correlation.max", "expert-output"]


class ExpertsGrouperForQwen2MoE(object):
    def __init__(
            self,
            config: Qwen2MoeConfig,
            similarity_fn: str = "cosine",
            similarity_base: str = "router-logits",
    ):
        if similarity_fn not in SIMILARITY_MAPPING_FUNCTION:
            raise ValueError(
                f"[MC-SMoE]similarity_fn should be one of {SIMILARITY_MAPPING_FUNCTION.keys()}, got {similarity_fn} instead."
            )
        if similarity_base not in LEGAL_SIMILARITY_BASES:
            raise ValueError(
                f"[MC-SMoE] similarity_base should be one of {LEGAL_SIMILARITY_BASES}, got {similarity_base} instead.")

        self.num_experts = config.num_experts
        self.d_model = config.hidden_size
        self.sparse_layer_indices = list(range(0, config.num_hidden_layers))
        self.similarity_fn = SIMILARITY_MAPPING_FUNCTION[similarity_fn]
        self.similarity_base = similarity_base
        self._group_state_dict = None
        self._similarity_state_dict = None
        self._usage_frequency_state_dict = None
        self.reset_all()

    def reset_all(self):
        if self.similarity_base == "mse":
            self.similarity_fn = SIMILARITY_MAPPING_FUNCTION["mse"]
            print("[MC-SMoE]Set similarity_fn to mse for mse similarity_base.")
        self._group_state_dict = dict()
        self._similarity_state_dict = dict()
        self._usage_frequency_state_dict = dict()
        # Similarity range: [0, 2]
        for layer_idx in self.sparse_layer_indices:
            ffn_name = f"model.layers.{layer_idx}.mlp"
            self._group_state_dict[ffn_name] = torch.arange(self.num_experts, device="cpu")
            self._similarity_state_dict[ffn_name] = torch.zeros(
                (self.num_experts, self.num_experts), device="cpu") + torch.eye(self.num_experts, device="cpu")
            self._usage_frequency_state_dict[ffn_name] = torch.zeros(self.num_experts, device="cpu")

    def similarity_state_dict(self) -> Dict[str, torch.Tensor]:
        return deepcopy(self._similarity_state_dict)

    def group_state_dict(self) -> Dict[str, torch.LongTensor]:
        return deepcopy(self._group_state_dict)

    def usage_frequency_state_dict(self) -> Dict[str, torch.Tensor]:
        return deepcopy(self._usage_frequency_state_dict)

    def save_similarity(self, mlp_name: str, i: int, j: int, similarity: float):
        self._similarity_state_dict[mlp_name][i, j] = similarity
        self._similarity_state_dict[mlp_name][j, i] = similarity

    def get_similarity(self, mlp_name: str, i: int, j: int) -> float:
        return self._similarity_state_dict[mlp_name][i, j].item()

    def get_similarity_matrix(self, mlp_name: str) -> torch.Tensor:
        return deepcopy(self._similarity_state_dict[mlp_name])

    def save_group_state_dict(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self._group_state_dict, os.path.join(save_dir, "group_state_dict.pt"))

    def load_group_state_dict(self, load_dir: str):
        self._group_state_dict = torch.load(os.path.join(load_dir, "group_state_dict.pt"))

    def _assign_num_groups_per_layer(
            self,
            num_average_groups: int,
            merging_layers: List[int],
    ) -> Dict[str, int]:
        num_grouping_layers = len(merging_layers)
        total_num_groups = num_average_groups * num_grouping_layers + self.num_experts * (
                len(self.sparse_layer_indices) - num_grouping_layers
        )
        all_usage_frequency = []
        usage_frequency_dict = deepcopy(self._usage_frequency_state_dict)
        for i, layer_idx in enumerate(self.sparse_layer_indices):
            ffn_name = f"model.layers.{layer_idx}.mlp"

            # 1. Experts in the excluded layers are always not merged.
            if layer_idx not in merging_layers:
                usage_frequency_dict[ffn_name] = torch.ones_like(usage_frequency_dict[ffn_name])

            # 2. Each layer must have at least one group, set the most used expert in a layer to frequency 1.
            max_usage_index = torch.argmax(usage_frequency_dict[ffn_name])
            usage_frequency_dict[ffn_name][max_usage_index] = 1.0

            # 3. Collect all usage frequency.
            all_usage_frequency.append(usage_frequency_dict[ffn_name])

        all_usage_frequency = torch.cat(all_usage_frequency, dim=0)
        sorted_usage_frequency, sorted_indices = torch.sort(all_usage_frequency, descending=True)
        num_groups_per_layer = dict()

        # Note: When threshold is 0.0, the actual number of groups is smaller than total_num_groups.
        if num_average_groups == self.num_experts:
            total_num_groups = total_num_groups - 1
        frequency_threshold = sorted_usage_frequency[total_num_groups]
        print(f"[MC-SMoE] Frequency threshold: {frequency_threshold}")

        if frequency_threshold == 1.0:
            raise ValueError("[MC-SMoE] The number of groups is too large, please reduce the number of groups.")

        for i, layer_idx in enumerate(self.sparse_layer_indices):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            num_groups_per_layer[ffn_name] = torch.sum(
                (usage_frequency_dict[ffn_name] > frequency_threshold).long()
            ).item()

        return num_groups_per_layer

    def group_experts_randomly(
        self,
        num_groups: int,
    ):
        for layer_idx in tqdm(self.sparse_layer_indices,
                              desc=f"Randomly merging experts into {num_groups} clusters"):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            group_labels = generate_random_group_labels(self.num_experts, num_groups)
            self._group_state_dict[ffn_name] = group_labels
    
    def group_experts_globally_from_dominant_experts(
            self,
            num_average_groups: int,
            merging_layers: List[int],
    ) -> Dict[str, List[int]]:
        """
        Globally group experts into clusters by routing-guided clustering, each layer will have different number of
         clusters. The total number of clusters is determined by num_average_groups.

        Parameters
        ----------
        num_average_groups: int
            The average number of clusters for all layers.
        merging_layers: List[int]
            The layers of decoder that are excluded from merging.

        Returns
        -------
        core_experts: Dict[str, List[int]]
            The core experts of each cluster
        """

        # 1. Assign num_groups respectively for each layer according to num_average_groups
        num_groups_per_layer = self._assign_num_groups_per_layer(
            num_average_groups, merging_layers
        )
        print(f"[MC-SMoE] Number of groups per layer: {num_groups_per_layer}")

        # 2. Group experts into clusters for each layer
        dom_experts = dict()
        for layer_idx in tqdm(
                self.sparse_layer_indices,
                desc=f"[MC-SMoE] Globally routing-guided grouping experts into average {num_average_groups} clusters"
        ):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            num_groups = num_groups_per_layer[ffn_name]
            indices_sorted_by_usage = torch.argsort(self._usage_frequency_state_dict[ffn_name], descending=True)

            # 1 Assign top-K most-used experts with label 0 to K-1 respectively
            core_expert_indices = indices_sorted_by_usage[:num_groups]
            dom_experts[ffn_name] = core_expert_indices.tolist()
            for i in range(num_groups):
                self._group_state_dict[ffn_name][indices_sorted_by_usage[i]] = i

            # 2 Assign left unassigned experts to the cluster with the most similar core
            similarity_matrix = self.get_similarity_matrix(ffn_name)
            for i in range(num_groups, self.num_experts):
                # Find the most similar core
                most_similar_core = core_expert_indices[
                    torch.argmax(similarity_matrix[i, core_expert_indices])
                ]
                most_similar_group_label = self._group_state_dict[ffn_name][most_similar_core]
                self._group_state_dict[ffn_name][i] = most_similar_group_label

        return dom_experts

    def compute_all_usages(
            self,
            model: Qwen2MoeForCausalLM,
            dataloader: DataLoader,
    ):
        model.eval()
        config = model.config
        for batch in tqdm(dataloader, desc=f"[MC-SMoE] Evaluating routing distribution"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                # We don't need to compute loss here, so remove the labels
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch, output_router_logits=True)
            all_router_logits = outputs.router_logits
            all_router_logits = torch.stack(all_router_logits)  # of shape (num_hidden_layers, num_tokens, num_experts)
            selected_experts = torch.topk(all_router_logits, 2, dim=-1)[1].reshape(
                config.num_hidden_layers, -1
            )  # of shape (num_hidden_layers, num_tokens * 2)
            for layer_idx in self.sparse_layer_indices:
                ffn_name = f"model.layers.{layer_idx}.mlp"
                unique, counts = torch.unique(selected_experts[layer_idx], return_counts=True)
                self._usage_frequency_state_dict[ffn_name][unique.cpu()] += counts.cpu()
        self._usage_frequency_state_dict = {
            k: v / torch.sum(v) for k, v in self._usage_frequency_state_dict.items()
        }

    def compute_all_similarities(
            self,
            model: Qwen2MoeForCausalLM,
            dataloader: DataLoader = None
    ):
        similarity_list = ["weight", "router-weight", "router-logits", "expert-output"]
        if self.similarity_base not in similarity_list and dataloader is None:
            raise ValueError(
                "[MC-SMoE] `dataloader` should be provided when similarity_base is not 'weight' or 'router-weight'")
        model = model.eval()
        if self.similarity_base == "weight":
            self._compute_all_similarities_by_weight(model.state_dict())
        elif self.similarity_base == 'router-weight':
            self._compute_all_similarities_by_router_weight(model.state_dict())
        elif self.similarity_base == 'router-logits':
            self._compute_all_similarities_by_router_logits(model, dataloader)
        elif self.similarity_base == 'expert-output':
            self._compute_all_similarities_by_expert_outputs(model, dataloader)
        else:
            raise NotImplementedError

    def _compute_all_similarities_by_weight(self, state_dict: Dict[str, torch.Tensor]):
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[MC-SMoE]  Computing similarities by weight..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            for i in range(self.num_experts):
                for j in range(i + 1, self.num_experts):
                    i_flat = torch.cat(
                        [state_dict[f"{ffn_name}.experts.{i}.gate_proj.weight"].flatten(),
                         state_dict[f"{ffn_name}.experts.{i}.down_proj.weight"].flatten(),
                         state_dict[f"{ffn_name}.experts.{i}.up_proj.weight"].flatten()],
                        dim=0
                    )
                    j_flat = torch.cat(
                        [state_dict[f"{ffn_name}.experts.{j}.gate_proj.weight"].flatten(),
                         state_dict[f"{ffn_name}.experts.{j}.down_proj.weight"].flatten(),
                         state_dict[f"{ffn_name}.experts.{j}.up_proj.weight"].flatten()],
                        dim=0
                    )
                    similarity = self.similarity_fn(i_flat, j_flat)
                    self.save_similarity(ffn_name, i, j, similarity)

    def _compute_all_similarities_by_router_weight(
            self, state_dict: Dict[str, torch.Tensor]
    ):
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[MC-SMoE] Computing similarities by router rows..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            for i in range(self.num_experts):
                for j in range(i + 1, self.num_experts):
                    i_flat = state_dict[f"{ffn_name}.gate.weight"][i]
                    j_flat = state_dict[f"{ffn_name}.gate.weight"][j]
                    similarity = self.similarity_fn(i_flat, j_flat)
                    self.save_similarity(ffn_name, i, j, similarity)

    def _compute_all_similarities_by_router_logits(
            self, model: Qwen2MoeForCausalLM, dataloader: DataLoader
    ):
        model.eval()
        all_router_logits = []
        for batch in tqdm(dataloader, desc=f"[MC-SMoE] Running inference to get routing logits"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                # We don't need to compute loss here, so remove the labels
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch, output_router_logits=True)
            batch_router_logits = outputs.router_logits
            batch_router_logits = torch.stack(batch_router_logits)  # (num_hidden_layers, num_tokens, num_experts)
            all_router_logits.append(batch_router_logits)

        all_router_logits = torch.cat(all_router_logits, dim=1)  # (num_hidden_layers, *, num_experts)
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[MC-SMoE] Computing similarities by router logits..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            layer_router_logits = all_router_logits[layer_idx].reshape(-1, self.num_experts)
            with torch.no_grad():
                for i in range(self.num_experts):
                    for j in range(i + 1, self.num_experts):
                        i_flat = layer_router_logits[:, i].flatten()
                        j_flat = layer_router_logits[:, j].flatten()
                        similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(ffn_name, i, j, similarity)
    
    def _compute_all_similarities_by_expert_outputs(
            self, model: Qwen2MoeForCausalLM, dataloader: DataLoader
    ):
        model.eval()
        forwarded_hidden_states = {} # moe input
        handles = []
        def _get_activation_hook(name):
            def hook(module, input, output):
                # forwarded_hidden_states[name].append(input[0].detach().cpu().reshape(-1, input[0].shape[-1]))
                forwarded_hidden_states[name].append(input[0].detach().reshape(-1, input[0].shape[-1]))
            return hook
        
        for layer_idx in tqdm(
                self.sparse_layer_indices,
                desc=f"[Merging]Registering forward hook..."
        ):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            forwarded_hidden_states[ffn_name] = []
            handles.append(model.model.layers[layer_idx].mlp.register_forward_hook(
                _get_activation_hook(ffn_name))
            )

        for batch in tqdm(dataloader, desc=f"[MC-SMoE] Running inference to collect moe inputs"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                # We don't need to compute loss here, so remove the labels
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch)
                del outputs
        
        for handle in handles:
            handle.remove()
        torch.cuda.empty_cache()

        for layer_idx in tqdm(self.sparse_layer_indices, desc="[MC-SMoE] Computing similarities by expert outputs..."):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            layer_input = torch.cat(forwarded_hidden_states[ffn_name]).cuda()
            expert_outputs = [] # (E, #T, D) -> average -> (E, D)
            with torch.no_grad():
                for i in range(self.num_experts):
                    expert_outputs.append(model.model.layers[layer_idx].mlp.experts[i](layer_input).mean(dim=0))
                for i in range(self.num_experts):
                    for j in range(self.num_experts):
                        i_flat = expert_outputs[i].flatten()
                        j_flat = expert_outputs[j].flatten()
                        similarity = self.similarity_fn(i_flat, j_flat)
                        self.save_similarity(ffn_name, i, j, similarity)
        
        torch.cuda.empty_cache()

@torch.no_grad()
def _merge_mlp_experts_by_usage_frequency_weighting(
        ffn: Qwen2MoeSparseMoeBlock,
        group_labels: torch.LongTensor,
        usage_frequencies: torch.Tensor,
) -> Qwen2MoeSparseMoeBlock:
    for label in group_labels.unique():
        expert_indices = torch.where(group_labels == label)[0]
        gate_proj_weight_list = torch.stack(
            [ffn.experts[expert_idx].gate_proj.weight * usage_frequencies[expert_idx]
             for expert_idx in expert_indices], dim=0
        )
        down_proj_weight_list = torch.stack(
            [ffn.experts[expert_idx].down_proj.weight * usage_frequencies[expert_idx]
             for expert_idx in expert_indices], dim=0
        )
        up_proj_weight_list = torch.stack(
            [ffn.experts[expert_idx].up_proj.weight * usage_frequencies[expert_idx]
             for expert_idx in expert_indices], dim=0
        )
        gate_proj_weight = torch.sum(gate_proj_weight_list, dim=0) / (torch.sum(usage_frequencies[expert_indices], dim=0) + FP32_EPS)
        down_proj_weight = torch.sum(down_proj_weight_list, dim=0) / (torch.sum(usage_frequencies[expert_indices], dim=0) + FP32_EPS)
        up_proj_weight = torch.sum(up_proj_weight_list, dim=0) / (torch.sum(usage_frequencies[expert_indices], dim=0) + FP32_EPS)

        ffn.experts[expert_indices[0]].gate_proj.weight.copy_(gate_proj_weight)
        ffn.experts[expert_indices[0]].down_proj.weight.copy_(down_proj_weight)
        ffn.experts[expert_indices[0]].up_proj.weight.copy_(up_proj_weight)

        for expert_idx in expert_indices[1:]:
            # Binding merged experts to the first of them
            ffn.experts[expert_idx] = ffn.experts[expert_indices[0]]

    return ffn

@torch.no_grad()
def merge_qwen_moe_by_activation_matching_within_and_across_models(
    ffn_list: List[Qwen2MoeMLP],
    forwarded_hidden_states: torch.Tensor,
    mini_batch_size: Optional[int] = None,
    alpha_for_repeated_merging: Optional[float] = 0.1,
    average_coefs: Optional[List[float]] = None,
    input_weight: Optional[List[float]] = None,
) -> Qwen2MoeMLP:
    
    ffn_list = [ffn.eval() for ffn in ffn_list]
    concat_ffn = deepcopy(ffn_list[0])
    d_ff, d_model = concat_ffn.gate_proj.out_features, concat_ffn.gate_proj.in_features
    if input_weight is not None:
        average_coefs = []
        for w in input_weight:
            coef = [w] * d_ff
            average_coefs.extend(coef)
    elif average_coefs is None:
        average_coefs = [1.0] * len(ffn_list) * d_ff
    elif len(average_coefs) == len(ffn_list):
        average_coefs = [coef for coef in average_coefs for _ in range(d_ff)]
    elif len(average_coefs) != len(ffn_list) * d_ff:
        raise ValueError(
            f"The length of average_coefs should be either {len(ffn_list)} or {len(ffn_list) * d_ff}, "
            f"but got {len(average_coefs)}."
        )
    num_ffn = len(ffn_list)
    if len(forwarded_hidden_states) == 0 or len(forwarded_hidden_states) == 1:
        return concat_ffn
    if mini_batch_size is None:
        mini_batch_size = forwarded_hidden_states.shape[0]

    ffn_all_gate_proj = torch.cat([ffn.gate_proj.weight.data for ffn in ffn_list], dim=0)
    ffn_all_down_proj = torch.cat([ffn.down_proj.weight.data for ffn in ffn_list], dim=1)
    ffn_all_up_proj = torch.cat([ffn.up_proj.weight.data for ffn in ffn_list], dim=0)
    concat_ffn.gate_proj = torch.nn.Linear(d_model, d_ff * num_ffn, bias=False)
    concat_ffn.down_proj = torch.nn.Linear(d_ff * num_ffn, d_model, bias=False)
    concat_ffn.up_proj = torch.nn.Linear(d_model, d_ff * num_ffn, bias=False)
    concat_ffn.gate_proj.weight.data = ffn_all_gate_proj
    concat_ffn.down_proj.weight.data = ffn_all_down_proj
    concat_ffn.up_proj.weight.data = ffn_all_up_proj
    concat_ffn = concat_ffn.eval().to(forwarded_hidden_states.device)
    
    activations = []
    
    def _activation_hook(module, input, output):
        activations.append(input[0].detach().reshape(-1, input[0].shape[-1]))
        return _activation_hook
    
    print(f"Collect activations with batch size {mini_batch_size} with original data length {forwarded_hidden_states.shape[0]}")

    handle = concat_ffn.down_proj.register_forward_hook(_activation_hook)

    for i in range(0, forwarded_hidden_states.shape[0], mini_batch_size):
        concat_ffn(forwarded_hidden_states[i:i + mini_batch_size])
    
    handle.remove()
    del handle, forwarded_hidden_states

    # Modified version for moving data to cpu
    """
    activations_dict = {}
    handles = []
    def get_hook(name):
        def _activation_hook(module, input, output):
            # activations.append(input[0].detach().reshape(-1, input[0].shape[-1]))
            activations_dict[name].append(input[0].detach().cpu().reshape(-1, input[0].shape[-1]))
        return _activation_hook
    
    # handle = concat_ffn.down_proj.register_forward_hook(_activation_hook)

    print(f"Collect activations with batch size {mini_batch_size} with original data length {forwarded_hidden_states.shape[0]}")
    randperm_indices = torch.randperm(forwarded_hidden_states.shape[0])
    forwarded_hidden_states = forwarded_hidden_states[randperm_indices[:10000]] # forwarded_hidden_states.shape[0] // 40
    forwarded_hidden_states = forwarded_hidden_states.cuda()

    for ffn_idx, ffn in enumerate(ffn_list):
        ffn = ffn.to(forwarded_hidden_states.device)
        activations_dict[ffn_idx] = []
        handles.append(ffn.down_proj.register_forward_hook(get_hook(ffn_idx)))

    
    with torch.no_grad():
        for i in range(0, forwarded_hidden_states.shape[0], mini_batch_size):
            # concat_ffn(forwarded_hidden_states[i:i + mini_batch_size])
            for ffn_idx, ffn in enumerate(ffn_list):
                ffn(forwarded_hidden_states[i:i + mini_batch_size])

    for handle in handles:
        handle.remove()
    del handles, forwarded_hidden_states

    activations = []
    for i in range(len(activations_dict[0])):
        concat_tensor = torch.cat([activations_dict[k][i] for k in range(len(activations_dict))], dim=1)
        activations.append(concat_tensor)

    activations = torch.cat(activations, dim=0)  # (batch_size * seq_len, d_ff * num_ffn)
    del activations_dict
    for ffn in ffn_list:
        ffn = ffn.cpu()
    print("activations: ", activations.shape)
    """

    activations = torch.cat(activations, dim=0)  # (batch_size * seq_len, d_ff * num_ffn)

    # Initialize the correlation matrix
    mean = activations.mean(dim=0, keepdim=True)  # (1, d_ff * num_ffn)
    std = activations.std(dim=0, keepdim=True)  # (1, d_ff * num_ffn)
    covar = torch.mm(
        (activations - mean).t(),
        (activations - mean)
    ) / (activations.shape[0] - 1)  # (d_ff * num_ffn, d_ff * num_ffn)
    corr_matrix = covar / (std.t() * std + FP32_EPS)  # (d_ff * num_ffn, d_ff * num_ffn)

    del activations, covar, std, mean
    torch.cuda.empty_cache()

    corr_matrix[torch.arange(d_ff * num_ffn), torch.arange(d_ff * num_ffn)] = -1  # Remove self-correlation

    # Greedy Merging!
    while ffn_all_gate_proj.shape[0] > d_ff:
        # Select the most correlated pair
        max_index = torch.argmax(corr_matrix)
        max_i, max_j = max_index // corr_matrix.shape[0], max_index % corr_matrix.shape[0]

        # Merge the most correlated pair, replace the first feature with the merged one
        i_coef, j_coef = average_coefs[max_i], average_coefs[max_j]
        ffn_all_gate_proj[max_i] = (i_coef * ffn_all_gate_proj[max_i] + j_coef * ffn_all_gate_proj[max_j]) / (i_coef + j_coef + FP32_EPS)
        ffn_all_up_proj[max_i] = (i_coef * ffn_all_up_proj[max_i] + j_coef * ffn_all_up_proj[max_j]) / (i_coef + j_coef + FP32_EPS)
        ffn_all_down_proj[:, max_i] = (i_coef * ffn_all_down_proj[:, max_i] + j_coef * ffn_all_down_proj[:, max_j]) / (
                i_coef + j_coef + FP32_EPS)
       
        # Remove the second feature
        ffn_all_gate_proj = torch.cat([
            ffn_all_gate_proj[:max_j],
            ffn_all_gate_proj[max_j + 1:]
        ], dim=0)
        ffn_all_up_proj = torch.cat([
            ffn_all_up_proj[:max_j],
            ffn_all_up_proj[max_j + 1:]
        ], dim=0)
        ffn_all_down_proj = torch.cat([
            ffn_all_down_proj[:, :max_j],
            ffn_all_down_proj[:, max_j + 1:]
        ], dim=1)

        # Update the correlation matrix
        updated_corr_vec = alpha_for_repeated_merging * torch.min(
            torch.stack([corr_matrix[max_i], corr_matrix[max_j]]), dim=0
        ).values
        corr_matrix[max_i] = updated_corr_vec
        corr_matrix[:, max_i] = updated_corr_vec
        corr_matrix[max_i, max_i] = -1  # Remove self-correlation

        # Remove the second feature from the correlation matrix
        corr_matrix = torch.cat([
            corr_matrix[:, :max_j],
            corr_matrix[:, max_j + 1:]
        ], dim=1)
        corr_matrix = torch.cat([
            corr_matrix[:max_j],
            corr_matrix[max_j + 1:]
        ], dim=0)

        # Update the average coefs
        average_coefs[max_i] += average_coefs[max_j]
        average_coefs = average_coefs[:max_j] + average_coefs[max_j + 1:]

    # handle.remove()
    del corr_matrix
    merged_ffn = deepcopy(ffn_list[0])
   
    merged_ffn.gate_proj.weight.data = ffn_all_gate_proj
    merged_ffn.down_proj.weight.data = ffn_all_down_proj
    merged_ffn.up_proj.weight.data = ffn_all_up_proj

    return merged_ffn

@torch.no_grad()
def _merge_moe_experts_within_and_across_models(
        moe: Qwen2MoeSparseMoeBlock,
        group_labels: torch.LongTensor,
        forwarded_hidden_states: Tuple[torch.Tensor],
        dominant_alone: bool,
        mode: Optional[str] = "normal",
        core_expert_indices: Optional[List[int]] = None,
        usage_frequencies: Optional[torch.Tensor] = None,
) -> Qwen2MoeSparseMoeBlock:

    moe.expert_dict = {}
    input_weight = None

    for label in group_labels.unique():
        expert_indices = torch.where(group_labels == label)[0]

        if mode == "input-weight":
            input_weight = []
            for expert_idx in expert_indices:
                input_weight.append(forwarded_hidden_states[expert_idx].shape[0])
            s = sum(input_weight)
            input_weight = [w / s for w in input_weight]
            # input_weight /= sum(input_weight)

        # not dominant
        group_forwarded_hidden_states = torch.cat([
            forwarded_hidden_states[expert_idx] for expert_idx in expert_indices
        ], dim=0)
        if len(expert_indices) == 1:
            merged_expert = moe.experts[expert_indices[0]]
        else:
            merged_expert = merge_qwen_moe_by_activation_matching_within_and_across_models(
                ffn_list=[moe.experts[expert_idx] for expert_idx in expert_indices],
                forwarded_hidden_states=group_forwarded_hidden_states,
                mini_batch_size=2048,
                average_coefs=usage_frequencies[expert_indices].tolist() if usage_frequencies is not None else None,
                input_weight=input_weight,
            )
        moe.experts[expert_indices[0]].gate_proj.weight.copy_(merged_expert.gate_proj.weight)
        moe.experts[expert_indices[0]].down_proj.weight.copy_(merged_expert.down_proj.weight)
        moe.experts[expert_indices[0]].up_proj.weight.copy_(merged_expert.up_proj.weight)

        moe.expert_dict[expert_indices[0].item()] = expert_indices[0].item()

        for expert_idx in expert_indices[1:]:
            # Binding merged experts to the first of them
            # moe.experts[expert_idx] = moe.experts[expert_indices[0]]
            moe.expert_dict[expert_idx.item()] = expert_indices[0].item()
            moe.experts[expert_idx.item()] = None
    print(moe.expert_dict)
    moe.forward = MethodType(merged_qwen2moe_forward, moe)
    return moe

@torch.no_grad()
def merge_by_groups_with_usage_weighted(
        model: Qwen2MoeForCausalLM,
        grouper: ExpertsGrouperForQwen2MoE,
        merging_layers: Optional[List[int]] = None,
) -> Qwen2MoeForCausalLM:
    usage_frequency_dict = grouper.usage_frequency_state_dict()
    group_labels_dict = grouper.group_state_dict()

    for layer_idx in tqdm(
            grouper.sparse_layer_indices,
            desc=f"[MC-SMoE] Merging experts with usage-frequency-weighted averaging..."
    ):
        if merging_layers is not None and layer_idx not in merging_layers:
            continue
        ffn_name = f"model.layers.{layer_idx}.mlp"
        group_labels = group_labels_dict[ffn_name]
        usage_frequencies = usage_frequency_dict[ffn_name]
        model.model.layers[layer_idx].mlp = _merge_mlp_experts_by_usage_frequency_weighting(
            ffn=model.model.layers[layer_idx].mlp,
            group_labels=group_labels,
            usage_frequencies=usage_frequencies,
        )
    return model


@torch.no_grad()
def merge_by_groups_within_and_across_models(
    qwen_model: Qwen2MoeForCausalLM,
    grouper: ExpertsGrouperForQwen2MoE,
    dataloader: DataLoader,
    mode: Optional[str] = "normal",
    partition: Optional[int] = 2,
    dominant_alone: Optional[bool] = False,
    core_experts: Optional[Dict[str, List[int]]] = None,
    usage_weighted: Optional[bool] = False,
) -> Qwen2MoeForCausalLM:
    
    forwarded_hidden_states = dict()

    usage_frequencies = grouper.usage_frequency_state_dict()
    num_experts = grouper.num_experts

    def part_processor(sparse_layer_indices):
        qwen_model.eval() #.cuda()
        handles = []

        def _get_activation_hook(name):
            def hook(module, input, output):
                # forwarded_hidden_states[name].append(input[0].detach().cpu().reshape(-1, input[0].shape[-1]))
                forwarded_hidden_states[name].append(input[0].detach().reshape(-1, input[0].shape[-1]))
            return hook
        
        for layer_idx in tqdm(
                sparse_layer_indices,
                desc=f"[Merging]Registering forward hook..."
        ):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            forwarded_hidden_states[ffn_name] = []
            handles.append(qwen_model.model.layers[layer_idx].mlp.register_forward_hook(
                _get_activation_hook(ffn_name))
            )
        
        router_indices = {name: [] for name in forwarded_hidden_states.keys()}
        if mode == "activation-with-router-logits":
            router_weights = {name: [] for name in forwarded_hidden_states.keys()}
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="[Merging]Computing activations..."):
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = qwen_model(**batch, output_router_logits=True)
                for layer_idx in sparse_layer_indices:
                    ffn_name = f"model.layers.{layer_idx}.mlp"
                    routing_weights = F.softmax(outputs.router_logits[layer_idx], dim=1, dtype=torch.float)
                    routing_weights, selected_experts = torch.topk(routing_weights, qwen_model.config.num_experts_per_tok, dim=-1)
                    router_indices[ffn_name].append(selected_experts)
                    if mode == "activation-with-router-logits":
                        router_weights[ffn_name].append(routing_weights)
                        
        for handle in handles:
            handle.remove()
        
        
        for layer_idx in tqdm(
                sparse_layer_indices,
                desc=f"[Merging]Merging by groups within and across experts..."
        ):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            group_labels = grouper.group_state_dict()[ffn_name]
            layer_forwarded_hidden_states = tuple()
            for expert_idx in range(num_experts): # expert num
                hidden_states_list = []
                for i in range(len(dataloader)): # batch of data
                    batch_tensor = torch.tensor([False for _ in range(len(forwarded_hidden_states[ffn_name][i]))])
                    if mode == "activation-with-router-logits":
                        router_weight = []
                        for j in range(len(forwarded_hidden_states[ffn_name][i])): # one token
                            for r, ind in enumerate(router_indices[ffn_name][i][j]): # token's router-logits and expert-index
                                if expert_idx == ind:
                                    batch_tensor[j] = True
                                    router_weight.append(router_weights[ffn_name][i][j][r])
                        # router_weight = torch.tensor(router_weight).unsqueeze(1).cpu().to(forwarded_hidden_states[ffn_name][i].dtype)
                        router_weight = torch.tensor(router_weight).unsqueeze(1).to(forwarded_hidden_states[ffn_name][i])
                        hidden_states_list.append(forwarded_hidden_states[ffn_name][i][batch_tensor] * router_weight)
                    else:
                        for j in range(len(forwarded_hidden_states[ffn_name][i])): # one token
                            if expert_idx in router_indices[ffn_name][i][j]:
                                batch_tensor[j] = True
                        hidden_states_list.append(forwarded_hidden_states[ffn_name][i][batch_tensor])
                layer_forwarded_hidden_states += (
                    torch.cat(hidden_states_list, dim=0),
                )
            qwen_model.model.layers[layer_idx].mlp = _merge_moe_experts_within_and_across_models(
                moe=qwen_model.model.layers[layer_idx].mlp,
                group_labels=group_labels,
                forwarded_hidden_states=layer_forwarded_hidden_states,
                dominant_alone=dominant_alone,
                mode=mode,
                core_expert_indices=core_experts[ffn_name] if core_experts is not None else None,
                usage_frequencies=usage_frequencies[ffn_name] if usage_weighted else None,
            )

    print(grouper.sparse_layer_indices)
    partition_num = len(grouper.sparse_layer_indices) // partition
    for i in range(0, len(grouper.sparse_layer_indices), partition_num):
        cur_indices = grouper.sparse_layer_indices[i:i+partition_num]
        print("cur: ", cur_indices)
        part_processor(cur_indices)
        print(torch.cuda.memory_summary())
    return qwen_model
