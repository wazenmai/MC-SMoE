import os
import pickle
import time
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
from mcsmoe.models.qwen import merged_qwen2moe_forward, Qwen2MoEWrapper

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
            start_layer: int = 0,
            group_limit: int = 4,
            data_limit: int = 1000000,
    ):
        if similarity_fn not in SIMILARITY_MAPPING_FUNCTION:
            raise ValueError(
                f"[TAMP]similarity_fn should be one of {SIMILARITY_MAPPING_FUNCTION.keys()}, got {similarity_fn} instead."
            )
        if similarity_base not in LEGAL_SIMILARITY_BASES:
            raise ValueError(
                f"[TAMP] similarity_base should be one of {LEGAL_SIMILARITY_BASES}, got {similarity_base} instead.")

        self.num_experts = config.num_experts
        self.d_model = config.hidden_size
        self.d_ff = config.moe_intermediate_size
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.group_limit = group_limit
        self.data_limit = data_limit

        self.sparse_layer_indices = list(range(start_layer, config.num_hidden_layers))
        self.similarity_fn = SIMILARITY_MAPPING_FUNCTION[similarity_fn]
        self.similarity_base = similarity_base
        self._group_state_dict = None
        self._similarity_state_dict = None
        self._usage_frequency_state_dict = None
        self.reset_all()

    def reset_all(self):
        if self.similarity_base == "mse":
            self.similarity_fn = SIMILARITY_MAPPING_FUNCTION["mse"]
            print("[TAMP]Set similarity_fn to mse for mse similarity_base.")
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
        print("total_num_groups: ", total_num_groups)
        all_usage_frequency = []
        usage_frequency_dict = deepcopy(self._usage_frequency_state_dict)
        for i, layer_idx in enumerate(self.sparse_layer_indices):
            ffn_name = f"model.layers.{layer_idx}.mlp"

            # 1. Experts in the excluded layers are always not merged.
            if layer_idx not in merging_layers:
                usage_frequency_dict[ffn_name] = torch.ones_like(usage_frequency_dict[ffn_name])

            # 2. Each layer must have at least one group, set the most used expert in a layer to frequency 1.
            k = (self.num_experts // self.group_limit) + 1 if (self.num_experts % self.group_limit) != 0 else (self.num_experts // self.group_limit)
            # max_usage_index = torch.argmax(usage_frequency_dict[ffn_name])
            # usage_frequency_dict[ffn_name][max_usage_index] = 1.0

            # 3. Collect all usage frequency.
            all_usage_frequency.append(usage_frequency_dict[ffn_name])

        all_usage_frequency = torch.cat(all_usage_frequency, dim=0)
        sorted_usage_frequency, sorted_indices = torch.sort(all_usage_frequency, descending=True)
        num_groups_per_layer = dict()

        # Note: When threshold is 0.0, the actual number of groups is smaller than total_num_groups.
        if num_average_groups == self.num_experts:
            total_num_groups = total_num_groups - 1
        frequency_threshold = sorted_usage_frequency[total_num_groups]
        print(f"[TAMP] Frequency threshold: {frequency_threshold}")

        if frequency_threshold == 1.0:
            raise ValueError("[TAMP] The number of groups is too large, please reduce the number of groups.")

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
        print(f"[TAMP] Number of groups per layer: {num_groups_per_layer}")

        # 2. Group experts into clusters for each layer
        dom_experts = dict()
        for layer_idx in tqdm(
                self.sparse_layer_indices,
                desc=f"[TAMP] Globally routing-guided grouping experts into average {num_average_groups} clusters"
        ):
            ffn_name = f"model.layers.{layer_idx}.mlp"
            num_groups = num_groups_per_layer[ffn_name]
            group_member_count = torch.zeros(num_groups)

            indices_sorted_by_usage = torch.argsort(self._usage_frequency_state_dict[ffn_name], descending=True)

            # 1 Assign top-K most-used experts with label 0 to K-1 respectively
            group_dict = {} 
            core_expert_indices = indices_sorted_by_usage[:num_groups]
            dom_experts[ffn_name] = core_expert_indices.tolist()
            for i in range(num_groups):
                self._group_state_dict[ffn_name][indices_sorted_by_usage[i]] = i
                group_member_count[i] += 1
                group_dict[i] = [core_expert_indices[i].item()]
            # 2 Assign left unassigned experts to the cluster with the most similar core
            similarity_matrix = self.get_similarity_matrix(ffn_name)
            print(similarity_matrix)
            print(core_expert_indices)
            for i in range(0, self.num_experts):
                if i in core_expert_indices:
                    continue
                # Find the most similar core
                most_similar_core = core_expert_indices[
                    torch.argmax(similarity_matrix[i, core_expert_indices])
                ]
                most_similar_group_label = self._group_state_dict[ffn_name][most_similar_core]
                self._group_state_dict[ffn_name][i] = most_similar_group_label
                group_member_count[most_similar_group_label] += 1
                group_dict[most_similar_group_label.item()].append(i)
                print(f"--expert {i} is assigned to group {most_similar_group_label}, the core expert is {most_similar_core}")
                if group_member_count[self._group_state_dict[ffn_name][i]] > self.group_limit:
                    if len(core_expert_indices) == 1 and self.group_limit < self.num_experts:
                        raise ValueError(
                            f"[Merging]The number of groups at Encoder layer {layer_idx} is too small!"
                        )
                    
                    while group_member_count[most_similar_group_label] > self.group_limit:
                        print(f"----meet group limit {self.group_limit} with group {most_similar_group_label} (core: {most_similar_core})")
                        # Find the most unsimilar expert in the exceed group
                        sim = similarity_matrix[most_similar_core, group_dict[most_similar_group_label.item()]]
                        print(sim, group_dict[most_similar_group_label.item()])
                        unsimilar_pos = torch.argmin(sim).item()
                        if (unsimilar_pos == 0): # do not let it choose the dominant expert
                            unsimilar_pos = 1
                        unsimilar_idx = group_dict[most_similar_group_label.item()][unsimilar_pos]
                    
                        group_member_count[most_similar_group_label] -= 1
                        group_dict[most_similar_group_label.item()].remove(unsimilar_idx)
                        similarity_matrix[unsimilar_idx, most_similar_core] = -1
                        similarity_matrix[most_similar_core, unsimilar_idx] = -1
                        print(f"----kick out {unsimilar_idx} from group ")
                        # Reassign group label
                        most_similar_core = core_expert_indices[
                            torch.argmax(similarity_matrix[unsimilar_idx, core_expert_indices])
                        ]
                        most_similar_group_label = self._group_state_dict[ffn_name][most_similar_core]
                        self._group_state_dict[ffn_name][unsimilar_idx] = most_similar_group_label
                        group_member_count[most_similar_group_label] += 1
                        group_dict[most_similar_group_label.item()].append(unsimilar_idx)
                        print(f"--expert {unsimilar_idx} is assigned to group {most_similar_group_label}, the core expert is {most_similar_core}")
        return dom_experts

    def compute_all_usages(
            self,
            model: Qwen2MoeForCausalLM,
            dataloader: DataLoader,
    ):
        model.eval()
        config = model.config
        for batch in tqdm(dataloader, desc=f"[TAMP] Evaluating routing distribution"):
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
        # if os.path.exists("similarity.pkl"):
        #     with open("similarity.pkl", "rb") as f:
        #         self._similarity_state_dict = pickle.load(f)
        #     return
        
        similarity_list = ["weight", "router-weight", "router-logits", "expert-output"]
        if self.similarity_base not in similarity_list and dataloader is None:
            raise ValueError(
                "[TAMP] `dataloader` should be provided when similarity_base is not 'weight' or 'router-weight'")
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
        
        # with open("similarity.pkl", "wb") as f:
        #     pickle.dump(self._similarity_state_dict, f)

    def _compute_all_similarities_by_weight(self, state_dict: Dict[str, torch.Tensor]):
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[TAMP]  Computing similarities by weight..."):
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
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[TAMP] Computing similarities by router rows..."):
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
        for batch in tqdm(dataloader, desc=f"[TAMP] Running inference to get routing logits"):
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
        for layer_idx in tqdm(self.sparse_layer_indices, desc="[TAMP] Computing similarities by router logits..."):
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

        for batch in tqdm(dataloader, desc=f"[TAMP] Running inference to collect moe inputs"):
            batch = {k: v.cuda() for k, v in batch.items()}
            if "labels" in batch:
                # We don't need to compute loss here, so remove the labels
                batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch)
                del outputs
        
        for handle in handles:
            handle.remove()
        # torch.cuda.empty_cache()

        for layer_idx in tqdm(self.sparse_layer_indices, desc="[TAMP] Computing similarities by expert outputs..."):
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
        
        # torch.cuda.empty_cache()
    
    def all_in_one_knowledge_dominant(
            self,
            model: Qwen2MoeForCausalLM,
            dataloader: DataLoader,
            merge: Optional[str] = "zipit",
            mode: Optional[str] = "normal",
            num_groups: Optional[int] = 30,
            dominant_alone: Optional[bool] = False,
            usage_weighted: Optional[bool] = False,
            lam_pred: Optional[float] = 1.0,
            lam_rep: Optional[float] = 1e-5,
            T: Optional[float] = 2.0,
    ):
        forwarded_hidden_states = [] # collect on cpu
        core_experts = dict()
        kd_labels = [] # collect on cpu

        #TODO: collect kd outputs
        def _get_activation_hook(name):
            def hook(module, input, output):
                forwarded_hidden_states.append(input[0].detach().cpu().reshape(-1, input[0].shape[-1]))
            return hook
        
        for layer_idx in self.sparse_layer_indices:
            _st = time.time()
            moe = model.model.layers[layer_idx].mlp
            _device = moe.experts[0].gate_proj.weight.device
            _dtype = moe.experts[0].gate_proj.weight.dtype
            moe_name = f"model.layers.{layer_idx}.mlp"
            print(f"[Process-Start] === Layer {layer_idx} / {len(self.sparse_layer_indices)} -> {moe_name} ===")

            # STEP: 1. Compute similarity
            print(self._similarity_state_dict[moe_name][0])
            
            # STEP: 2. Compute knowledge + Collect activation for zipit merging
            # 2.1 Initialization
            model.eval()
            # model.zero_grad()
            for p in model.parameters():
                p.requires_grad_(False)
            moe_pred_kl = torch.zeros(self.num_experts, self.d_ff, device=_device)
            moe_rep_kl = torch.zeros(self.num_experts, self.d_ff, device=_device)
            moe_masks = torch.ones(self.num_experts, self.d_ff, dtype=_dtype, device=_device)
            moe_masks.requires_grad_(True)

            # Zipit variables
            router_indices = []
            router_weights = []

            # 2.2 Register hook function
            handles = []
            _inputs = {}
            moe = model.model.layers[layer_idx].mlp
            experts = model.model.layers[layer_idx].mlp.experts
            for e in range(self.num_experts):
                handles.append(
                    apply_mask(experts[e].down_proj, moe_masks[e])
                )
                _inputs[e] = []
                handles.append(
                    hijack(experts[e].down_proj, _inputs[e], _hijack_input=True)
                )
            handles.append(moe.register_forward_hook(
                _get_activation_hook(moe_name)
            ))

            # 2.3 Do forward and measure knowledge
            num_samples = 0
            num_tokens = 0
            _index = 0
            print("[Computing] Do forward and measure knowledge on batch")
            for b, batch in enumerate(dataloader):
                print(b, end='')
                batch = {k: v.to(_device) for k, v in batch.items()}
                att_mask = batch["attention_mask"].bool().reshape(-1)
                batch_samples = batch["attention_mask"].shape[0]
                num_samples += batch_samples
                num_tokens += att_mask.sum().item()
                outputs = model(**batch, output_router_logits=True)
                router_logits = outputs.router_logits

                if layer_idx == self.sparse_layer_indices[0]:
                    pred = F.softmax(outputs.logits / T, dim=1).detach()
                    kd_labels.append(pred.cpu())
                else:
                    pred = kd_labels[_index:_index + batch_samples, :].to(_device)
                    _index += batch_samples
                kl_div = F.kl_div(
                    input=F.log_softmax(outputs.logits / T, dim=1),
                    target=pred,
                    reduction="batchmean"
                ) * (T ** 2)
                if kl_div > 10:
                    kl_div /= 1000
                print(f"kl_div: {kl_div.item()}")

                kl_div.backward()

                del outputs, pred, kl_div
                
                # Measure amount of knowledge
                routing_weights = F.softmax(router_logits[layer_idx], dim=1)
                routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
                router_indices.append(selected_experts)
                if mode == "activation-with-router-logits" or mode == "all":
                    if hasattr(model.config, "norm_topk_prob"):
                        if model.config.norm_topk_prob:
                            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
                    else:
                        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
                    router_weights.append(routing_weights)
                expert_index = selected_experts[att_mask]
                del routing_weights, selected_experts
                for e in range(self.num_experts):
                    # get feature
                    # token_id = (expert_index == e).nonzero()
                    # number_of_tokens = token_id.shape[0]
                    # _features = _inputs[e][-1][:number_of_tokens].to(torch.float32).to(_device)
                    _features = _inputs[e][-1].to(torch.float32).to(_device)

                    # get weigth and calculate representational knowledge
                    _weight = experts[e].down_proj.weight.detach()
                    moe_rep_kl[e] += ((_features ** 2).sum(dim=0) * (_weight ** 2).mean(dim=0)).data

                    # get gradient and calculate predictive knowledge
                    grad = moe_masks.grad[e]
                    moe_pred_kl[e] += (grad.detach() ** 2) * 0.5

                    del _inputs[e][-1], _features, _weight, grad
                moe_masks.grad.zero_()
            print(torch.cuda.memory_summary())
            # 2.4 Average score
            moe_pred_kl /= num_samples
            moe_rep_kl /= num_tokens

            print(f"moe_pred_kl: {moe_pred_kl.shape} {moe_pred_kl}")
            print(f"moe_rep_kl: {moe_rep_kl.shape} {moe_rep_kl}")

            # 2.5 Compute score
            origin_moe_scores = (moe_rep_kl * lam_rep + moe_pred_kl * lam_pred)
            moe_scores = origin_moe_scores.mean(dim=-1) # (E,)
            print(f"moe_scores: {moe_scores.shape} {moe_scores}")

            if layer_idx == self.sparse_layer_indices[0]:
                kd_labels = torch.cat(kd_labels, dim=0)
                print(f"kd_labels: {kd_labels.shape} {kd_labels}")
            
            for handle in handles:
                handle.remove()
            del handles, moe_pred_kl, moe_rep_kl, moe_masks, _inputs

            # STEP: 3. Choose dominant experts by knowledge, group them by similarity
            print(f"[Process-Grouping] === Choose dominant experts by knowledge, group them by similarity {self.similarity_base} ===")
            group_member_count = torch.zeros(num_groups)
            indices_sorted_by_kl = torch.argsort(moe_scores, descending=True).cpu()
            core_expert_indices = indices_sorted_by_kl[:num_groups]
            print("core_expert_indices: ", core_expert_indices)
            core_experts[moe_name] = core_expert_indices.tolist()
            group_dict = {}          
            for i in range(num_groups):
                self._group_state_dict[moe_name][core_expert_indices[i]] = i
                group_member_count[i] += 1
                group_dict[i] = [core_expert_indices[i].item()]
            similarity_matrix = self.get_similarity_matrix(moe_name)
            for i in range(0, self.num_experts): # assign group label to left experts
                if i in core_expert_indices:
                    continue
                most_similar_core = core_expert_indices[
                    torch.argmax(similarity_matrix[i, core_expert_indices])
                ]
                most_similar_group_label = self._group_state_dict[moe_name][most_similar_core]
                self._group_state_dict[moe_name][i] = most_similar_group_label
                group_member_count[most_similar_group_label] += 1
                group_dict[most_similar_group_label.item()].append(i)
                print(f"--expert {i} is assigned to group {most_similar_group_label}, the core expert is {most_similar_core}")
                if group_member_count[self._group_state_dict[moe_name][i]] > self.group_limit:
                    if len(core_expert_indices) == 1 and self.group_limit < self.num_experts:
                        raise ValueError(
                            f"[Merging]The number of groups at Encoder layer {layer_idx} is too small!"
                        )
                    
                    while group_member_count[most_similar_group_label] > self.group_limit:
                        print(f"----meet group limit {self.group_limit} with group {most_similar_group_label} (core: {most_similar_core})")
                        # Find the most unsimilar expert in the exceed group
                        sim = similarity_matrix[most_similar_core, group_dict[most_similar_group_label.item()]]
                        unsimilar_idx = group_dict[most_similar_group_label.item()][torch.argmin(sim).item()]
                        print(sim, group_dict[most_similar_group_label.item()])
                        unsimilar_pos = torch.argmin(sim).item()
                        if (unsimilar_pos == 0): # do not let it choose the dominant expert
                            unsimilar_pos = 1
                        unsimilar_idx = group_dict[most_similar_group_label.item()][unsimilar_pos]
                        group_member_count[most_similar_group_label] -= 1
                        group_dict[most_similar_group_label.item()].remove(unsimilar_idx)
                        similarity_matrix[unsimilar_idx, most_similar_core] = -1
                        similarity_matrix[most_similar_core, unsimilar_idx] = -1
                        print(f"----kick out {unsimilar_idx} from group ")
                        # Reassign group label
                        most_similar_core = core_expert_indices[
                            torch.argmax(similarity_matrix[unsimilar_idx, core_expert_indices])
                        ]
                        most_similar_group_label = self._group_state_dict[moe_name][most_similar_core]
                        self._group_state_dict[moe_name][unsimilar_idx] = most_similar_group_label
                        group_member_count[most_similar_group_label] += 1
                        group_dict[most_similar_group_label.item()].append(unsimilar_idx)
                        print(f"--expert {unsimilar_idx} is assigned to group {most_similar_group_label}, the core expert is {most_similar_core}")

            # STEP: 4. Zipit Merge
            group_labels = self._group_state_dict[moe_name]
            layer_forwarded_hidden_states = tuple()
            forwarded_hidden_states = torch.cat(forwarded_hidden_states, dim=0) # T x D
            cat_router_indices = torch.cat(router_indices, dim=0) # BT x k
            if mode == "activation-with-router-logits" or mode == "all":
                cat_router_weights = torch.cat(router_weights, dim=0) # BT x k
            for e in range(self.num_experts):
                expert_mask = (cat_router_indices == e)
                batch_tensor = torch.any(expert_mask, dim=-1).to(forwarded_hidden_states.device)
                choice_input = forwarded_hidden_states[batch_tensor]
                if mode == "activation-with-router-logits" or mode == "all":
                    router_weight = torch.masked_select(cat_router_weights, expert_mask).view(-1, 1).to(choice_input.device)
                    hidden_states = choice_input * router_weight
                else:
                    hidden_states = choice_input
                layer_forwarded_hidden_states += (hidden_states,)
            model.model.layers[layer_idx].mlp = _merge_moe_experts_within_and_across_models(
                moe=moe,
                group_labels=group_labels,
                forwarded_hidden_states=layer_forwarded_hidden_states,
                dominant_alone=dominant_alone,
                merge=merge,
                mode=mode,
                core_expert_indices=core_experts[moe_name] if core_experts is not None else None,
                usage_frequencies=self._usage_frequency_state_dict[moe_name] if usage_weighted else None,
                moe_scores=origin_moe_scores,
                data_limit=self.data_limit,
            )

            del layer_forwarded_hidden_states
            forwarded_hidden_states = []
            print(f"[Process-End] === Layer {layer_idx} / {len(self.sparse_layer_indices)}, cost {time.time() - _st:2f}s ===")
        self.core_experts = core_experts
        return model

def apply_mask(module, _mask):
    # applying masks to the input to compute gradients
    def masking(_, i):
        return _mask * i[0]

    handle = module.register_forward_pre_hook(masking)
    return handle

def hijack(module, _list, _hijack_input, _stop_forward=False):
    # if _stop_forward=True, then it raise error after forwarding the module
    if _hijack_input:
        def input_hook(_, inputs, __):
            _list.append(inputs[0].detach().cpu()) # .clone().data
            if _stop_forward:
                raise StopFowardException

        handle = module.register_forward_hook(input_hook)
    else:
        raise NotImplementedError
    return handle         

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
def _merge_qwen_moe_by_knowledge_weight(
    ffn_list: List[Qwen2MoeMLP],
    knowledge_weight: torch.Tensor,
) -> Qwen2MoeMLP:
    d_ff, d_model = ffn_list[0].gate_proj.out_features, ffn_list[0].gate_proj.in_features
    num_ffn = len(ffn_list)
    
    col_sum = knowledge_weight.sum(dim=0, keepdim=True)
    knowledge_weight = (knowledge_weight / col_sum)
    knowledge = knowledge_weight.reshape(1, -1) # (ExN) -> (1xEN)
    
    print(knowledge_weight.shape, knowledge.shape)
    print(knowledge_weight)

    ffn_all_gate_proj = knowledge.T * torch.cat([ffn.gate_proj.weight.data for ffn in ffn_list], dim=0).to(knowledge.dtype)
    ffn_all_down_proj = knowledge * torch.cat([ffn.down_proj.weight.data for ffn in ffn_list], dim=1).to(knowledge.dtype)
    ffn_all_up_proj = knowledge.T * torch.cat([ffn.up_proj.weight.data for ffn in ffn_list], dim=0).to(knowledge.dtype)
    
    print(ffn_all_gate_proj.shape)

    ffn_all_gate_proj = ffn_all_gate_proj.reshape(d_ff, num_ffn, d_model)
    ffn_all_down_proj = ffn_all_down_proj.reshape(d_model, num_ffn, d_ff)
    ffn_all_up_proj = ffn_all_up_proj.reshape(d_ff, num_ffn, d_model)

    ffn_gate_proj = ffn_all_gate_proj.sum(dim=1)
    ffn_down_proj = ffn_all_down_proj.sum(dim=1)
    ffn_up_proj = ffn_all_up_proj.sum(dim=1)

    merged_ffn = deepcopy(ffn_list[0])
    merged_ffn.gate_proj.weight.data = ffn_gate_proj.to(ffn_list[0].gate_proj.weight.dtype)
    merged_ffn.down_proj.weight.data = ffn_down_proj.to(ffn_list[0].down_proj.weight.dtype)
    merged_ffn.up_proj.weight.data = ffn_up_proj.to(ffn_list[0].up_proj.weight.dtype)
    return merged_ffn

def remove_col(x, idx):
    return torch.cat([x[:, :idx], x[:, idx+1:]], dim=-1)

def remove_row(x, idx):
    return torch.cat([x[:idx], x[idx+1:]], dim=0)

@torch.no_grad()
def _zipit_merge(temp_dim, target_dim, weight1, weight3, data, _device, _dtype):
    permutation_matrix = torch.eye(temp_dim, temp_dim, dtype=_dtype, device=_device)
    ROUND = 0
    act = torch.nn.SiLU()
    while temp_dim > target_dim:
        ROUND += 1
        odd = temp_dim % 2
        target_dim_this_round = max(target_dim, temp_dim // 2 + odd)
        print(f"ROUND {ROUND}. From {temp_dim} to {target_dim_this_round}")
        
        ### Collect activations
        activations = []
        if weight3 is None:
            cur = torch.matmul(data, weight1.T)
        else:
            cur = act(torch.matmul(data, weight1.T)) * torch.matmul(data, weight3.T)
        activations.append(cur.reshape(-1, cur.shape[-1]))
        activations = torch.cat(activations, dim=0)
        print("Activations: ", activations.shape)
        ### Compute covariance
        mean = activations.mean(dim=0, keepdim=True)
        std = activations.std(dim=0, keepdim=True)
        covar = torch.matmul((activations - mean).T, activations - mean) / (activations.shape[0] - 1)
        corr_matrix = covar / (std.T * std + FP32_EPS)
        del mean, std, covar
        torch.cuda.empty_cache()
        corr_matrix[torch.arange(temp_dim), torch.arange(temp_dim)] = -1 # Remove self-correlation
        print(corr_matrix)
        ### Merge temp_dim / 2 times
        for _ in range(temp_dim - target_dim_this_round):
            max_index = torch.argmax(corr_matrix)
            row, col = max_index // corr_matrix.shape[0], max_index % corr_matrix.shape[0]
            permutation_matrix[:, row] += permutation_matrix[:, col]
            permutation_matrix = remove_col(permutation_matrix, col)

            # row_coef, col_coef = average_coefs[row], average_coefs[col]
            row_coef, col_coef = 1.0, 1.0
            weight1[row] = (row_coef * weight1[row] + col_coef * weight1[col]) / (row_coef + col_coef + FP32_EPS)
            if weight3 is not None:
                weight3[row] = (row_coef * weight3[row] + col_coef * weight3[col]) / (row_coef + col_coef + FP32_EPS)
                weight3 = remove_row(weight3, col)
            weight1 = remove_row(weight1, col)
            
            corr_matrix[row] = FP32_EPS # set very small number to avoid repeated merging
            corr_matrix[:, row] = FP32_EPS
            corr_matrix[row, row] = -1
            corr_matrix = remove_col(corr_matrix, col)
            corr_matrix = remove_row(corr_matrix, col)
        temp_dim = weight1.shape[0]
    for i in range(20): # permutation_matrix.shape[1]
        print(permutation_matrix[:, i].nonzero().squeeze())
    return permutation_matrix

@torch.no_grad()
def _merge_qwen_moe_by_zipit(
    ffn_list: List[Qwen2MoeMLP],
    forwarded_hidden_states: torch.Tensor,
    mini_batch_size: Optional[int] = None,
    alpha_for_repeated_merging: Optional[float] = 0.1,
    average_coefs: Optional[List[float]] = None,
    input_weight: Optional[List[float]] = None,
) -> Qwen2MoeMLP:
    d_ff, d_model = ffn_list[0].gate_proj.out_features, ffn_list[0].gate_proj.in_features
    num_ffn = len(ffn_list)
    temp_dim = d_ff * num_ffn
    average_coefs = [1.0] * temp_dim
    act = torch.nn.SiLU()

    _device = ffn_list[0].gate_proj.weight.device
    _dtype = ffn_list[0].gate_proj.weight.dtype
    forwarded_hidden_states = forwarded_hidden_states.to(_device)
    print(f"Data shape: {forwarded_hidden_states.shape}, temp_dim: {temp_dim}, target_dim: {d_ff}")

    ### Merge gate_proj and up_proj
    ffn_all_gate_proj = torch.cat([ffn.gate_proj.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_all_up_proj = torch.cat([ffn.up_proj.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    first_permutation_matrix = _zipit_merge(d_ff * num_ffn, d_ff, ffn_all_gate_proj, ffn_all_up_proj, forwarded_hidden_states, _device, _dtype)
    first_unmerge_matrix = first_permutation_matrix
    first_merge_matrix = torch.div(first_permutation_matrix, torch.sum(first_permutation_matrix, dim=0, keepdim=True))

    ffn_all_gate_proj = torch.cat([ffn.gate_proj.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_all_up_proj = torch.cat([ffn.up_proj.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_gate_proj = torch.matmul(first_merge_matrix.T, ffn_all_gate_proj)
    ffn_up_proj = torch.matmul(first_merge_matrix.T, ffn_all_up_proj)

    ### Merge down_proj
    new_data = act(torch.matmul(forwarded_hidden_states, ffn_gate_proj.T)) * torch.matmul(forwarded_hidden_states, ffn_up_proj.T)
    ffn_all_down_proj = torch.cat([ffn.down_proj.weight.data for ffn in ffn_list], dim=0) # (d_model * num_ffn, d_ff)
    second_permutation_matrix = _zipit_merge(d_model * num_ffn, d_model, ffn_all_down_proj, None, new_data, _device, _dtype)
    second_merge_matrix = torch.div(second_permutation_matrix, torch.sum(second_permutation_matrix, dim=0, keepdim=True))
    second_unmerge_matrix = second_permutation_matrix
    ffn_down_proj = torch.zeros(d_model, d_ff).to(_device)
    for i in range(num_ffn):
        ffn_down_proj += torch.matmul(second_merge_matrix.T[:, i*d_model:(i+1)*d_model], torch.matmul(ffn_all_down_proj[i*d_model:(i+1)*d_model], first_unmerge_matrix.T[:, i*d_ff:(i+1)*d_ff]))

    merged_ffn = deepcopy(ffn_list[0])
    merged_ffn.gate_proj.weight.data = ffn_gate_proj
    merged_ffn.down_proj.weight.data = ffn_down_proj
    merged_ffn.up_proj.weight.data = ffn_up_proj

    return merged_ffn

@torch.no_grad()
def collect_act(data, weight1, weight3=None):
    activations = []
    act = torch.nn.SiLU()
    if weight3 is not None:
        cur = act(torch.matmul(data, weight1.T)) * torch.matmul(data, weight3.T)
    else:
        cur = torch.matmul(data, weight1.T)
    activations.append(cur.reshape(-1, cur.shape[-1]))
    return torch.cat(activations, dim=0)

@torch.no_grad()
def compute_covariance(act1, act2):
    mean1 = act1.mean(dim=0, keepdim=True)
    mean2 = act2.mean(dim=0, keepdim=True)
    std1 = act1.std(dim=0, keepdim=True)
    std2 = act2.std(dim=0, keepdim=True)
    corr_matrix = torch.matmul((act1 - mean1).T, act2 - mean2) / (act1.shape[0] - 1)
    corr_matrix = corr_matrix / (std1.T * std2 + FP32_EPS)
    return corr_matrix

def compute_permutation(data, ffn_list, dom_ind):
    dom_act = collect_act(forwarded_hidden_states, ffn_list[dominant_index].gate_proj.weight.data, ffn_list[dominant_index].up_proj.weight.data)
    group_indexes = []
    for i in range(num_ffn):
        if i == dominant_index:
            continue
        other_act = collect_act(forwarded_hidden_states, ffn_list[i].gate_proj.weight.data, ffn_list[i].up_proj.weight.data)
        corr_matrix = compute_covariance(dom_act, other_act)
        max_index = torch.argmax(corr_matrix, dim=1)
        group_indexes.append(max_index)
    
    permutation_matrix = torch.eye(d_ff, temp_dim).to(_device).to(torch.float16)
    for i in range(d_ff):
        for j in range(corr_matrix.shape[1]):
            index_in_this_group = (group_indexes[j] == i).nonzero().squeeze() + d_ff * (i + 1)
            permutation_matrix[i, index_in_this_group] = 1
    permutation_matrix = torch.div(permutation_matrix, torch.sum(permutation_matrix, dim=1, keepdim=True))
    return permutation_matrix

@torch.no_grad()
def _merge_qwen_moe_experts_with_dominant(
        ffn_list: List[Qwen2MoeMLP],
        forwarded_hidden_states: torch.Tensor,
        mini_batch_size: Optional[int] = None,
        alpha_for_repeated_merging: Optional[float] = 0.1,
        average_coefs: Optional[List[float]] = None,
        input_weight: Optional[List[float]] = None,
        dominant_index: Optional[int] = 0,
) -> Qwen2MoeMLP:
    print("merge: fix-dom-independent-rule-without-unmerge")
    d_ff, d_model = ffn_list[0].gate_proj.out_features, ffn_list[0].gate_proj.in_features
    num_ffn = len(ffn_list)
    need_pinv = False
    if input_weight is not None:
        coef = input_weight
        need_pinv = True
    elif average_coefs is None:
        coef = [1.0] * num_ffn
    elif len(average_coefs) == num_ffn:
        coef = average_coefs
        need_pinv = True
    else:
        coef = [1.0] * num_ffn
    
    if dominant_index != 0:
        ffn_list[0], ffn_list[dominant_index] = ffn_list[dominant_index], ffn_list[0]
    print("dominant_index: ", dominant_index)
    _device = ffn_list[0].gate_proj.weight.device
    _dtype = ffn_list[0].gate_proj.weight.dtype
    forwarded_hidden_states = forwarded_hidden_states.to(_device)
    print(f"Data shape: {forwarded_hidden_states.shape}, temp_dim: {d_ff * num_ffn}, target_dim: {d_ff}, dominant_index: {dominant_index}")
    # Compute Permutation Matrix for gate_proj and up_proj
    permutation_matrix = torch.eye(d_ff, d_ff * num_ffn, device=_device, dtype=_dtype) * coef[0]
    dom_act = collect_act(forwarded_hidden_states, ffn_list[dominant_index].gate_proj.weight.data, ffn_list[dominant_index].up_proj.weight.data)
    group_indexes = []
    for i in range(num_ffn):
        if i == dominant_index:
            continue
        other_act = collect_act(forwarded_hidden_states, ffn_list[i].gate_proj.weight.data, ffn_list[i].up_proj.weight.data)
        corr_matrix = compute_covariance(dom_act, other_act)
        max_index = torch.argmax(corr_matrix, dim=1)
        group_indexes.append(max_index)
    for i in range(d_ff):
        for j in range(num_ffn - 1):
            index_in_this_group = (group_indexes[j] == i).nonzero().squeeze() + d_ff * (j + 1)
            permutation_matrix[i, index_in_this_group] = coef[j]
    if not need_pinv:
        unmerge_1 = permutation_matrix
        permutation_matrix = torch.div(permutation_matrix, torch.sum(permutation_matrix, dim=1, keepdim=True))
    else:
        permutation_matrix = torch.div(permutation_matrix, torch.sum(permutation_matrix, dim=1, keepdim=True))
        unmerge_1 = torch.linalg.pinv(permutation_matrix.to(torch.float)).to(_dtype).T
        permutation_matrix = permutation_matrix.to(_dtype)
    
    print(f"first permutation_matrix: {permutation_matrix.shape} {permutation_matrix[0]}")
    ffn_all_gate_proj = torch.cat([ffn.gate_proj.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_all_up_proj = torch.cat([ffn.up_proj.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_gate_proj = torch.matmul(permutation_matrix, ffn_all_gate_proj)
    ffn_up_proj = torch.matmul(permutation_matrix, ffn_all_up_proj)

    del ffn_all_gate_proj, ffn_all_up_proj

    # Compute Permutation Matrix for down_proj
    permutation_matrix = torch.eye(d_model, d_model * num_ffn, dtype=_dtype, device=_device) * coef[0]
    new_data = collect_act(forwarded_hidden_states, ffn_gate_proj, ffn_up_proj)
    dom_act = collect_act(new_data, ffn_list[dominant_index].down_proj.weight.data, None)
    group_indexes.clear()
    for i in range(num_ffn):
        if i == dominant_index:
            continue
        other_act = collect_act(new_data, ffn_list[i].down_proj.weight.data, None)
        corr_matrix = compute_covariance(dom_act, other_act)
        max_index = torch.argmax(corr_matrix, dim=1)
        group_indexes.append(max_index)
    for i in range(d_model):
        for j in range(num_ffn - 1):
            index_in_this_group = (group_indexes[j] == i).nonzero().squeeze() + d_model * (j + 1)
            permutation_matrix[i, index_in_this_group] = coef[j]
    permutation_matrix = torch.div(permutation_matrix, torch.sum(permutation_matrix, dim=1, keepdim=True))
    print(f"second permutation_matrix: {permutation_matrix.shape} {permutation_matrix[0]}")
    ffn_all_down_proj = torch.cat([ffn.down_proj.weight.data for ffn in ffn_list], dim=0) # (d_model * num_ffn, d_ff)
    ffn_down_proj = torch.zeros(d_model, d_ff).to(_device)
    for i in range(num_ffn):
        ffn_down_proj += torch.matmul(permutation_matrix[:, i*d_model:(i+1)*d_model],
            torch.matmul(ffn_all_down_proj[i*d_model:(i+1)*d_model], 
                         unmerge_1[:, i*d_ff:(i+1)*d_ff])
        )

    del ffn_all_down_proj

    merged_ffn = deepcopy(ffn_list[0])
    merged_ffn.gate_proj.weight.data = ffn_gate_proj
    merged_ffn.down_proj.weight.data = ffn_down_proj
    merged_ffn.up_proj.weight.data = ffn_up_proj

    return merged_ffn

@torch.no_grad()
def _merge_qwen_moe_experts_with_dominant_same_rule(
        ffn_list: List[Qwen2MoeMLP],
        forwarded_hidden_states: torch.Tensor,
        mini_batch_size: Optional[int] = None,
        alpha_for_repeated_merging: Optional[float] = 0.1,
        average_coefs: Optional[List[float]] = None,
        input_weight: Optional[List[float]] = None,
        dominant_index: Optional[int] = 0,
):
    print("merge: fix-dom-same-rule-without-unmerge")
    d_ff, d_model = ffn_list[0].gate_proj.out_features, ffn_list[0].gate_proj.in_features
    num_ffn = len(ffn_list)
    if input_weight is not None:
        coef = input_weight
    elif average_coefs is None:
        coef = [1.0] * num_ffn
    elif len(average_coefs) == num_ffn:
        coef = average_coefs
    else:
        coef = [1.0] * num_ffn
    
    if dominant_index != 0:
        ffn_list[0], ffn_list[dominant_index] = ffn_list[dominant_index], ffn_list[0]
    print("dominant_index: ", dominant_index)
    _device = ffn_list[0].gate_proj.weight.device
    _dtype = ffn_list[0].gate_proj.weight.dtype
    forwarded_hidden_states = forwarded_hidden_states.to(_device)
    print(f"Data shape: {forwarded_hidden_states.shape}, temp_dim: {d_ff * num_ffn}, target_dim: {d_ff}, dominant_index: {dominant_index}")
    # Compute Permutation Matrix for gate_proj and up_proj
    permutation_matrix = torch.eye(d_ff, d_ff * num_ffn, dtype=_dtype, device=_device) * coef[0]
    dom_act = collect_act(forwarded_hidden_states, ffn_list[dominant_index].gate_proj.weight.data, ffn_list[dominant_index].up_proj.weight.data)
    group_indexes = []
    for i in range(num_ffn):
        if i == dominant_index:
            continue
        other_act = collect_act(forwarded_hidden_states, ffn_list[i].gate_proj.weight.data, ffn_list[i].up_proj.weight.data)
        corr_matrix = compute_covariance(dom_act, other_act)
        max_index = torch.argmax(corr_matrix, dim=1)
        group_indexes.append(max_index)
    for i in range(d_ff):
        for j in range(num_ffn - 1):
            index_in_this_group = (group_indexes[j] == i).nonzero().squeeze() + d_ff * (j + 1)
            permutation_matrix[i, index_in_this_group] = coef[j]
    permutation_matrix = torch.div(permutation_matrix, torch.sum(permutation_matrix, dim=1, keepdim=True))
    print(f"first permutation_matrix: {permutation_matrix.shape} {permutation_matrix[0]}")
    ffn_all_gate_proj = torch.cat([ffn.gate_proj.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_all_down_proj = torch.cat([ffn.down_proj.weight.data for ffn in ffn_list], dim=1) # (d_model, d_ff * num_ffn)
    ffn_all_up_proj = torch.cat([ffn.up_proj.weight.data for ffn in ffn_list], dim=0) # (d_ff * num_ffn, d_model)
    ffn_gate_proj = torch.matmul(permutation_matrix, ffn_all_gate_proj)
    ffn_down_proj = torch.matmul(permutation_matrix, ffn_all_down_proj.T)
    ffn_up_proj = torch.matmul(permutation_matrix, ffn_all_up_proj)

    del ffn_all_gate_proj, ffn_all_down_proj, ffn_all_up_proj

    merged_ffn = deepcopy(ffn_list[0])
    merged_ffn.gate_proj.weight.data = ffn_gate_proj
    merged_ffn.down_proj.weight.data = ffn_down_proj.T
    merged_ffn.up_proj.weight.data = ffn_up_proj

    return merged_ffn

@torch.no_grad()
def process_coef(num_ffn, d_ff, d_model, average_coefs=None, input_weight=None):
    if input_weight is not None:
        first_coef = []
        second_coef = []
        for w in input_weight:
            coef_1 = [w] * d_ff
            first_coef.extend(coef_1)
            coef_2 = [w] * d_model
            second_coef.extend(coef_2)
    elif average_coefs is None:
        first_coef = [1.0] * num_ffn * d_ff
        second_coef = [1.0] * num_ffn * d_model
    elif len(average_coefs) == num_ffn:
        first_coef = [coef for coef in average_coefs for _ in range(d_ff)]
        second_coef = [coef for coef in average_coefs for _ in range(d_model)]
    else:
        raise ValueError("The argument `avearge_coefs` should be either None or have the same length as `num_ffn`, or you need to provide `input_weight`.")
    return first_coef, second_coef

@torch.no_grad()
def compute_correlation(act):
    mean = act.mean(dim=0, keepdim=True)
    std = act.std(dim=0, keepdim=True)
    covar = torch.mm((act - mean).T, act - mean) / (act.shape[0] - 1)
    corr_matrix = covar / (std.T * std + FP32_EPS)
    del mean, std, covar
    torch.cuda.empty_cache()
    return corr_matrix

@torch.no_grad()
def compute_merging(temp_dim, target_dim, corr_matrix, coef, alpha, _device):
    permutation_matrix = torch.eye(temp_dim, temp_dim, dtype=torch.float, device=_device)
    while corr_matrix.shape[0] > target_dim:
        max_index = torch.argmax(corr_matrix)
        max_i, max_j = max_index // corr_matrix.shape[0], max_index % corr_matrix.shape[0]

        # Update permutation matrix
        i_coef, j_coef = coef[max_i], coef[max_j]
        permutation_matrix[:, max_i] = (i_coef * permutation_matrix[:, max_i] + j_coef * permutation_matrix[:, max_j]) / (i_coef + j_coef + FP32_EPS)
        permutation_matrix = remove_col(permutation_matrix, max_j)

        # Update corr_matrix
        updated_corr_vec = alpha * torch.min(torch.stack([corr_matrix[max_i], corr_matrix[max_j]]), dim=0).values
        corr_matrix[max_i] = updated_corr_vec
        corr_matrix[:, max_i] = updated_corr_vec
        corr_matrix[max_i, max_i] = -1
        # Remove second feature from the correlation matrix
        corr_matrix = remove_col(corr_matrix, max_j)
        corr_matrix = remove_row(corr_matrix, max_j)
    return permutation_matrix

@torch.no_grad()
def _merge_qwen_moe_by_zipit_with_unmerge(
    ffn_list: List[Qwen2MoeMLP],
    forwarded_hidden_states: torch.Tensor,
    mini_batch_size: Optional[int] = 5000,
    alpha_for_repeated_merging: Optional[float] = 0.1,
    average_coefs: Optional[List[float]] = None,
    input_weight: Optional[List[float]] = None,
):
    print("merge: zipit-independe-rule-with-unmerge")
    ffn_list = [ffn.eval() for ffn in ffn_list]
    d_ff, d_model = ffn_list[0].gate_proj.out_features, ffn_list[0].gate_proj.in_features
    num_ffn = len(ffn_list)
    first_coef, second_coef = process_coef(num_ffn, d_ff, d_model, average_coefs, input_weight)
    
    _device = ffn_list[0].gate_proj.weight.device
    _dtype = ffn_list[0].gate_proj.weight.dtype
    forwarded_hidden_states = forwarded_hidden_states.to(_device)
    print(f"Collect activations with batch size {mini_batch_size} with original data length {forwarded_hidden_states.shape}")

    # Compute gate_proj and up_proj's permutation matrix
    ffn_all_gate_proj = torch.cat([ffn.gate_proj.weight.data for ffn in ffn_list], dim=0) # 
    ffn_all_up_proj = torch.cat([ffn.up_proj.weight.data for ffn in ffn_list], dim=0)
    act = torch.nn.SiLU()

    activations = []
    cur = act(torch.matmul(forwarded_hidden_states, ffn_all_gate_proj.T)) * torch.matmul(forwarded_hidden_states, ffn_all_up_proj.T)
    activations.append(cur.reshape(-1, cur.shape[-1]))
    cat_activtaions = torch.cat(activations, dim=0)
    activations.clear()
    corr_matrix = compute_correlation(cat_activtaions)
    corr_matrix[torch.arange(d_ff * num_ffn), torch.arange(d_ff * num_ffn)] = -1  # Remove self-correlation
    print(f"corr_matrix: {corr_matrix.shape}")
    first_permutation_matrix = compute_merging(d_ff * num_ffn, d_ff, corr_matrix, first_coef, alpha_for_repeated_merging, _device)
    first_permutation_matrix = first_permutation_matrix / torch.sum(first_permutation_matrix, dim=0, keepdim=True)
    first_unmerge_matrix = torch.linalg.pinv(first_permutation_matrix)
    first_permutation_matrix = first_permutation_matrix.to(_dtype)
    ffn_gate_proj = torch.matmul(first_permutation_matrix.T, ffn_all_gate_proj)
    ffn_up_proj = torch.matmul(first_permutation_matrix.T, ffn_all_up_proj)
    print(f"first_permutation_matrix: {first_permutation_matrix.shape}, first_unmerge_matrix: {first_unmerge_matrix.shape}")
    
    # Compute down_proj's permutation matrix
    ffn_all_down_proj = torch.cat([ffn.down_proj.weight.data for ffn in ffn_list], dim=0)
    new_data = act(torch.matmul(forwarded_hidden_states, ffn_gate_proj.T)) * torch.matmul(forwarded_hidden_states, ffn_up_proj.T)
    activations = []
    new_cur = torch.matmul(new_data, ffn_all_down_proj.T)
    activations.append(new_cur.reshape(-1, new_cur.shape[-1]))
    cat_activtaions = torch.cat(activations, dim=0)
    activations.clear()
    corr_matrix = compute_correlation(cat_activtaions)
    corr_matrix[torch.arange(d_model * num_ffn), torch.arange(d_model * num_ffn)] = -1  # Remove self-correlation
    print(f"corr_matrix: {corr_matrix.shape}")
    second_permutation_matrix = compute_merging(d_model * num_ffn, d_model, corr_matrix, second_coef, alpha_for_repeated_merging, _device)
    second_permutation_matrix = second_permutation_matrix / torch.sum(second_permutation_matrix, dim=0, keepdim=True)
    second_unmerge_matrix = torch.linalg.pinv(second_permutation_matrix) # DxED
    print(f"second_permutation_matrix: {second_permutation_matrix.shape}, second_unmerge_matrix: {second_unmerge_matrix.shape}")
    second_permutation_matrix = second_permutation_matrix.to(_device).to(_dtype)
    first_unmerge_matrix = first_unmerge_matrix.to(_device).to(_dtype)
    ffn_down_proj = torch.zeros(d_model, d_ff, device=_device)
    for i in range(num_ffn):
        ffn_down_proj += torch.matmul(second_permutation_matrix.T[:, i*d_model:(i+1)*d_model], 
            torch.matmul(ffn_all_down_proj[i*d_model:(i+1)*d_model], first_unmerge_matrix[:, i*d_ff:(i+1)*d_ff]))
    
    merged_ffn = deepcopy(ffn_list[0])
    merged_ffn.gate_proj.weight.data = ffn_gate_proj
    merged_ffn.down_proj.weight.data = ffn_down_proj
    merged_ffn.up_proj.weight.data = ffn_up_proj
    
    # TODO: use a warpper to warp moe and assign unmerge matrix to it
    # TODO: consider (gate_proj, up_proj) and (down_proj) has differnt unmerge matrix, use down_proj's unmerge matrix to unmerge down_proj's output
    return merged_ffn, second_unmerge_matrix

@torch.no_grad()
def _merge_qwen_moe_by_activation_matching_within_and_across_models(
    ffn_list: List[Qwen2MoeMLP],
    forwarded_hidden_states: torch.Tensor,
    mini_batch_size: Optional[int] = None,
    alpha_for_repeated_merging: Optional[float] = 0.1,
    average_coefs: Optional[List[float]] = None,
    input_weight: Optional[List[float]] = None,
) -> Qwen2MoeMLP:
    print("merge: zipit-same-rule-without-unmerge")
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

    _device = forwarded_hidden_states.device
    ffn_all_gate_proj = torch.cat([ffn.gate_proj.weight.data for ffn in ffn_list], dim=0)
    ffn_all_down_proj = torch.cat([ffn.down_proj.weight.data for ffn in ffn_list], dim=1)
    ffn_all_up_proj = torch.cat([ffn.up_proj.weight.data for ffn in ffn_list], dim=0)
    concat_ffn.gate_proj = torch.nn.Linear(d_model, d_ff * num_ffn, bias=False)
    concat_ffn.down_proj = torch.nn.Linear(d_ff * num_ffn, d_model, bias=False)
    concat_ffn.up_proj = torch.nn.Linear(d_model, d_ff * num_ffn, bias=False)
    concat_ffn.gate_proj.weight.data = ffn_all_gate_proj
    concat_ffn.down_proj.weight.data = ffn_all_down_proj
    concat_ffn.up_proj.weight.data = ffn_all_up_proj
    # concat_ffn = concat_ffn.eval()
    concat_ffn.gate_proj.weight.to(_device)
    concat_ffn.up_proj.weight.to(_device)
    concat_ffn.down_proj.weight.to(_device)
    
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
        merge: str,
        mode: Optional[str] = "normal",
        core_expert_indices: Optional[List[int]] = None,
        usage_frequencies: Optional[torch.Tensor] = None,
        moe_scores: Optional[torch.Tensor] = None,
        data_limit: Optional[int] = 50000,
) -> Qwen2MoeSparseMoeBlock:

    _device = moe.experts[0].gate_proj.weight.device
    _dtype = moe.experts[0].gate_proj.weight.dtype
    
    # moe.expert_dict = {}
    input_weight = None
    if merge == "unmerge":
        moe = Qwen2MoEWrapper(moe)
    print("core_expert_indices: ", core_expert_indices)

    for label in group_labels.unique():
        expert_indices = torch.where(group_labels == label)[0]
        print(f"Group {label}: {expert_indices}")
        core_expert_index = [i for i, idx in enumerate(expert_indices) if idx in core_expert_indices]
        if mode == "input-weight" or mode == "all":
            input_weight = []
            for expert_idx in expert_indices:
                input_weight.append(forwarded_hidden_states[expert_idx].shape[0])
            s = sum(input_weight)
            input_weight = [w / s for w in input_weight]
            print(f"Input weight: {input_weight}")
        # not dominant
        group_forwarded_hidden_states = torch.cat([
            forwarded_hidden_states[expert_idx] for expert_idx in expert_indices
        ], dim=0)
        if data_limit != 1000000:
            randperm_indices = torch.randperm(group_forwarded_hidden_states.shape[0])
            group_forwarded_hidden_states = group_forwarded_hidden_states[randperm_indices[:data_limit]]
        if len(expert_indices) == 1:
            if merge == "unmerge":
                merged_expert = moe.model.experts[expert_indices[0]]
                moe.unmerge_matrix[label.item()] = None
            else:
                merged_expert = moe.experts[expert_indices[0]]
        else:
            if merge == "kl-weight":
                merged_expert = _merge_qwen_moe_by_knowledge_weight(
                    ffn_list=[moe.experts[expert_idx] for expert_idx in expert_indices],
                    knowledge_weight=moe_scores[expert_indices],
                )
            elif merge == "update":
                merged_expert = _merge_qwen_moe_by_zipit(
                    ffn_list=[moe.experts[expert_idx] for expert_idx in expert_indices],
                    forwarded_hidden_states=group_forwarded_hidden_states,
                    mini_batch_size=5000,
                    average_coefs=usage_frequencies[expert_indices].tolist() if usage_frequencies is not None else None,
                    input_weight=input_weight,
                )
            elif merge == "fix-dom":
                merged_expert = _merge_qwen_moe_experts_with_dominant(
                    ffn_list=[moe.experts[expert_idx] for expert_idx in expert_indices],
                    forwarded_hidden_states=group_forwarded_hidden_states,
                    mini_batch_size=5000,
                    average_coefs=usage_frequencies[expert_indices].tolist() if usage_frequencies is not None else None,
                    input_weight=input_weight,
                    dominant_index=core_expert_index[0],
                )
            elif merge == "fix-dom-same":
                merged_expert = _merge_qwen_moe_experts_with_dominant_same_rule(
                    ffn_list=[moe.experts[expert_idx] for expert_idx in expert_indices],
                    forwarded_hidden_states=group_forwarded_hidden_states,
                    mini_batch_size=5000,
                    average_coefs=usage_frequencies[expert_indices].tolist() if usage_frequencies is not None else None,
                    input_weight=input_weight,
                    dominant_index=core_expert_index[0],
                )
            elif merge == "unmerge":
                merged_expert, unmerge_matrix = _merge_qwen_moe_by_zipit_with_unmerge(
                    ffn_list=[moe.model.experts[expert_idx] for expert_idx in expert_indices],
                    forwarded_hidden_states=group_forwarded_hidden_states,
                    mini_batch_size=5000,
                    average_coefs=usage_frequencies[expert_indices].tolist() if usage_frequencies is not None else None,
                    input_weight=input_weight,
                )
                moe.unmerge_matrix[label.item()] = unmerge_matrix.to(_device).to(_dtype)
            else:
                merged_expert = _merge_qwen_moe_by_activation_matching_within_and_across_models(
                    ffn_list=[moe.experts[expert_idx] for expert_idx in expert_indices],
                    forwarded_hidden_states=group_forwarded_hidden_states,
                    mini_batch_size=2048,
                    average_coefs=usage_frequencies[expert_indices].tolist() if usage_frequencies is not None else None,
                    input_weight=input_weight,
                )
        
        if merge == "unmerge":
            moe.model.experts[expert_indices[0].item()].gate_proj.weight.copy_(merged_expert.gate_proj.weight)
            moe.model.experts[expert_indices[0].item()].down_proj.weight.copy_(merged_expert.down_proj.weight)
            moe.model.experts[expert_indices[0].item()].up_proj.weight.copy_(merged_expert.up_proj.weight)
            moe.expert_to_group[expert_indices[0].item()] = label.item()
            moe.group_to_expert[label.item()] = [expert_indices[0].item()]
            for expert_idx in expert_indices[1:]:
                moe.model.experts[expert_idx.item()] = moe.model.experts[expert_indices[0].item()]
                moe.expert_to_group[expert_idx.item()] = label.item()
                moe.group_to_expert[label.item()].append(expert_idx.item())
            moe.group_to_expert[label.item()] = torch.tensor(moe.group_to_expert[label.item()])
        else:
            moe.experts[expert_indices[0]].gate_proj.weight.copy_(merged_expert.gate_proj.weight)
            moe.experts[expert_indices[0]].down_proj.weight.copy_(merged_expert.down_proj.weight)
            moe.experts[expert_indices[0]].up_proj.weight.copy_(merged_expert.up_proj.weight)

            for expert_idx in expert_indices[1:]:
                # Binding merged experts to the first of them
                moe.experts[expert_idx] = moe.experts[expert_indices[0]]
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
            desc=f"[TAMP] Merging experts with usage-frequency-weighted averaging..."
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
    merge: str,
    mode: Optional[str] = "normal",
    partition: Optional[int] = 1,
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
        if mode == "activation-with-router-logits" or mode == "all":
            router_weights = {name: [] for name in forwarded_hidden_states.keys()}
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="[Merging]Computing activations..."):
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = qwen_model(**batch, output_router_logits=True)
                for layer_idx in sparse_layer_indices:
                    ffn_name = f"model.layers.{layer_idx}.mlp"
                    routing_weights = F.softmax(outputs.router_logits[layer_idx], dim=1)
                    routing_weights, selected_experts = torch.topk(routing_weights, qwen_model.config.num_experts_per_tok, dim=-1)
                    router_indices[ffn_name].append(selected_experts)
                    if mode == "activation-with-router-logits" or mode == "all":
                        router_weights[ffn_name].append(routing_weights)
                        
        for handle in handles:
            handle.remove()
        
        
        for layer_idx in tqdm(
                sparse_layer_indices,
                desc=f"[Merging]Merging by groups within and across experts..."
        ):
            _st = time.time()
            ffn_name = f"model.layers.{layer_idx}.mlp"
            group_labels = grouper.group_state_dict()[ffn_name]
            layer_forwarded_hidden_states = tuple()
            hidden_states = torch.cat(forwarded_hidden_states[ffn_name], dim=0) # T x D
            concat_router_indices = torch.cat(router_indices[ffn_name], dim=0) # BT x k
            if mode == "activation-with-router-logits" or mode == "all":
                concat_router_weights = torch.cat(router_weights[ffn_name], dim=0) # BT x k
            for expert_idx in range(num_experts): # expert num
                expert_mask = (concat_router_indices == expert_idx)
                batch_tensor = torch.any(expert_mask, dim=-1).to(hidden_states.device)
                choice_input = hidden_states[batch_tensor]
                if mode == "activation-with-router-logits" or mode == "all":
                    router_weight = torch.masked_select(concat_router_weights, expert_mask).view(-1, 1).to(choice_input.device)
                    layer_hidden_states = choice_input * router_weight
                else:
                    layer_hidden_states = choice_input
                layer_forwarded_hidden_states += (layer_hidden_states,)
            qwen_model.model.layers[layer_idx].mlp = _merge_moe_experts_within_and_across_models(
                moe=qwen_model.model.layers[layer_idx].mlp,
                group_labels=group_labels,
                forwarded_hidden_states=layer_forwarded_hidden_states,
                dominant_alone=dominant_alone,
                merge=merge,
                mode=mode,
                core_expert_indices=core_experts[ffn_name] if core_experts is not None else None,
                usage_frequencies=usage_frequencies[ffn_name] if usage_weighted else None,
                data_limit=grouper.data_limit,
            )
            del layer_forwarded_hidden_states
            print(f"------- Layer {layer_idx} took {time.time() - _st:.2f}s -------\n")

    print(grouper.sparse_layer_indices)
    partition_num = len(grouper.sparse_layer_indices) // partition
    for i in range(0, len(grouper.sparse_layer_indices), partition_num):
        cur_indices = grouper.sparse_layer_indices[i:i+partition_num]
        print("cur: ", cur_indices)
        part_processor(cur_indices)
        print(torch.cuda.memory_summary())
    return qwen_model
