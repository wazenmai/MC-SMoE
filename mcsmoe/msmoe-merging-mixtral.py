# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/18
from typing import Optional
# from huggingface_hub import login

import os
import logging
import time
import torch
from fire import Fire
from transformers import MixtralForCausalLM, AutoTokenizer

from mcsmoe.evaluation import get_minipile_dataloder, evaluate_minipile_perplexity, evaluate_fewshot, get_calib_dataloder
from mcsmoe.merging.grouping_mixtral import ExpertsGrouperForMixtral, merge_by_groups_with_usage_weighted, merge_by_groups_within_and_across_models, check

logger = logging.getLogger(__name__)
# login(token="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")

def evaluate_mcsmoe(
        task: str,
        num_average_groups: int,
        model_name: Optional[str] = "mistralai/Mixtral-8x7B-v0.1",
        dominant: Optional[str] = "knowledge", # random, frequency, knowledge
        similarity_base: Optional[str] = "router-logits", # router-logits, weight, expert-output
        merge: Optional[str] = "zipit", # no, freq, zipit, update, fix-dom, unmerge, kl-weight
        mode: Optional[str] = "normal", # normal, activation-with-router-logits, input-weight, all
        n_sentences: Optional[int] = 32,
        train_batch_size: Optional[int] = 4,
        eval_batch_size: Optional[int] = 32,
        partition: Optional[int] = 1,
        start_layer: Optional[int] = 0,
        output_path: Optional[str] = None,
        result_path: Optional[str] = None,
        model_path: Optional[str] = None,
        group_limit: Optional[int] = 4,
        data_limit: Optional[int] = 50000,
        num_fewshot: Optional[int] = 0,
):
    print(f"Merge model {model_name} with {num_average_groups} group, {dominant} dominant + {similarity_base} grouping + {merge} {mode} merge, evaluate on {task}")
    
    # torch.cuda.memory._record_memory_history()
    # torch.set_printoptions(threshold=1000000)
    
    eval_ppl = (task == "minipile")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = MixtralForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.eval()

    if merge != "no":
        print("[TAMP] Number of parameters before merging:", model.num_parameters())

        dataloader_for_merging = get_calib_dataloder(
            dataset="c4",
            tokenizer=tokenizer,
            max_block_size=2048,
            n_blocks_for_stat=n_sentences, # 32, 128
            batch_size=train_batch_size,
            num_workers=4,
        )

        print(f"[TAMP] Merging into average {num_average_groups} groups...")
        group_st = time.time()

        grouper = ExpertsGrouperForMixtral(config=model.config, similarity_base=similarity_base, start_layer=start_layer, group_limit=group_limit, data_limit=data_limit)
        grouper.compute_all_similarities(model, dataloader_for_merging)
        
        # check(model, dataloader_for_merging, "teacher")

        if dominant == "random":
            grouper.group_experts_randomly(num_groups=num_average_groups)
            model = merge_by_groups_within_and_across_models(
                mixtral_model=model,
                grouper=grouper,
                dataloader=dataloader_for_merging,
                merge=merge,
                mode=mode,
                partition=partition,
                dominant_alone=False,
                usage_weighted=False
            )
            dom_experts = None
        elif dominant == "frequency":
            grouper.compute_all_usages(model, dataloader_for_merging)
            for k in grouper._usage_frequency_state_dict:
                for num in grouper._usage_frequency_state_dict[k]:
                    print(round(num.item(), 4), end=',')
                print()
            dom_experts = grouper.group_experts_globally_from_dominant_experts(
                num_average_groups=num_average_groups, merging_layers=list(range(start_layer, model.config.num_hidden_layers))
            )
            if merge == "freq":
                model = merge_by_groups_with_usage_weighted(
                    model, grouper=grouper, merging_layers=list(range(start_layer, model.config.num_hidden_layers))
                )
            else:
                model = merge_by_groups_within_and_across_models(
                    mixtral_model=model,
                    grouper=grouper,
                    dataloader=dataloader_for_merging,
                    merge=merge,
                    mode=mode,
                    partition=partition,
                    dominant_alone=False,
                    core_experts=dom_experts,
                    usage_weighted=False
                )
        elif dominant == "knowledge":
            model = grouper.all_in_one_knowledge_dominant(
                model=model, 
                dataloader=dataloader_for_merging,
                merge=merge,
                mode=mode,
                num_groups=num_average_groups,
                dominant_alone=False,
                usage_weighted=False,
            )
            dom_experts = grouper.core_experts
        else:
            raise ValueError(
                f"Accepted dominant methods are `random`, `frequency` and `knowledge`, but the input is `{dominant}`"
            )
        
        # check(model, dataloader_for_merging, "zipit")
        print(f"[TAMP] Merging time: {time.time() - group_st:2f} seconds")
        
        ### Print grouping results
        print(f"[TAMP] ========= Grouping results ========= ")
        for name, state in grouper.group_state_dict().items():
            if dom_experts is None:
                print(f"Group {name}: {state.tolist()}")
            else:
                print(f"Group {name}: {state.tolist()} (DOMs are {dom_experts[name]})")
        del grouper

        if merge == "unmerge":
            print(f"[TAMP] ======= Grouping of unmerge ======= ")
            for layer_idx in range(start_layer, model.config.num_hidden_layers):
                print(f"--- Layer {layer_idx} ---")
                print(f"expert_to_group: {model.model.layers[layer_idx].block_sparse_moe.expert_to_group}")
                print(f"group_to_expert: {model.model.layers[layer_idx].block_sparse_moe.group_to_expert}")
                print(f"unmerge_matrix: {model.model.layers[layer_idx].block_sparse_moe.unmerge_matrix}")

       

        ### Save the model
        print("[TAMP] Number of parameters after merging:", model.num_parameters())
        if num_average_groups < model.config.num_experts_per_tok:
            model.config.num_experts_per_tok = num_average_groups
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        torch.save(model.state_dict(), output_path+"/model.pth")

    if eval_ppl:
        evaluate_minipile_perplexity(
            model, tokenizer=tokenizer, batch_size=eval_batch_size, log=True
        )
    elif isinstance(task, str):
        evaluate_fewshot(
            model, tokenizer=tokenizer, task=task, num_fewshot=num_fewshot, output_path=result_path, log=True
        )
    else:
        for t in task:
            evaluate_fewshot(
                model, tokenizer=tokenizer, task=t, num_fewshot=num_fewshot, eval_batch_size=eval_batch_size, output_path=result_path, log=True
            )


if __name__ == "__main__":
    Fire(evaluate_mcsmoe)
