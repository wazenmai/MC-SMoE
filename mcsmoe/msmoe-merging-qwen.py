# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/18
import os
import gc
import time
from typing import Optional

import logging
import torch
from fire import Fire
from transformers import Qwen2MoeForCausalLM, AutoTokenizer

from mcsmoe.evaluation import get_minipile_dataloder, evaluate_minipile_perplexity, evaluate_fewshot, get_calib_dataloder
from mcsmoe.merging.grouping_qwen import ExpertsGrouperForQwen2MoE, merge_by_groups_with_usage_weighted, merge_by_groups_within_and_across_models

logger = logging.getLogger(__name__)

def evaluate_mcsmoe(
        task: str,
        num_average_groups: int,
        model_name: Optional[str] = "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        dominant: Optional[str] = "knowledge", # random, frequency, knowledge
        similarity_base: Optional[str] = "router-logits", # router-logits, weight, expert-output
        merge: Optional[str] = "zipit", # no, freq, zipit, update, fix-dom, unmerge, kl-weight, fix-dom-same
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
        data_limit: Optional[int] = 1000000,
        num_fewshot: Optional[int] = 0,
):
    print(f"Merge model {model_name} with {num_average_groups} group, {dominant} dominant + {similarity_base} grouping + {merge} merge - {mode}, evaluate on {task}")

    eval_ppl = (task == "minipile")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = Qwen2MoeForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    if model_path:
        model.load_state_dict(torch.load(model_name))
    model.eval()


    # TAMP!
    if merge != "no":
        dataloader_for_merging = get_calib_dataloder(
            dataset="c4",
            tokenizer=tokenizer,
            max_block_size=2048,
            n_blocks_for_stat=n_sentences, # 128, reduce size to avoid OOM
            batch_size=train_batch_size,
            num_workers=4,
        )

        print(f"[TAMP] Merging into average {num_average_groups} groups...")
        group_st = time.time()
        grouper = ExpertsGrouperForQwen2MoE(
            config=model.config,
            similarity_base=similarity_base,
            start_layer=start_layer,
            group_limit=group_limit,
            data_limit=data_limit,
        )
        grouper.compute_all_similarities(model, dataloader_for_merging)
        
        if dominant == "random":
            grouper.group_experts_randomly(num_groups=num_average_groups)
            # Freq-merge
            model = merge_by_groups_with_usage_weighted(
                model, grouper=grouper, merging_layers=list(range(start_layer, model.config.num_hidden_layers))
            )
        elif dominant == "frequency":
            grouper.compute_all_usages(model, dataloader_for_merging)
            dom_experts = grouper.group_experts_globally_from_dominant_experts(
                num_average_groups=num_average_groups, merging_layers=list(range(start_layer, model.config.num_hidden_layers))
            )
            if merge == "freq":
                model = merge_by_groups_with_usage_weighted(
                    model, grouper=grouper, merging_layers=list(range(start_layer, model.config.num_hidden_layers))
                )
            else:
                model = merge_by_groups_within_and_across_models(
                    qwen_model=model,
                    grouper=grouper,
                    dataloader=dataloader_for_merging,
                    merge=merge,
                    mode=mode,
                    partition=partition,
                    core_experts=dom_experts,
                    dominant_alone=False,
                    usage_weighted=False
                )
        elif dominant == "knowledge":
            model = grouper.all_in_one_knowledge_dominant(
                model=model, 
                dataloader=dataloader_for_merging, 
                merge=merge,
                mode=mode,
                num_groups=num_average_groups,
            )
            dom_experts = grouper.core_experts
        else:
            raise ValueError(f"Unknown dominant type: {dominant}")
        
        print(f"[TAMP] Merging time: {time.time() - group_st:.2f} seconds")

        print(f"[TAMP] ========= Grouping results ========= ")
        for name, state in grouper.group_state_dict().items():
            if dom_experts is None:
                print(f"Group {name}: {state.tolist()}")
            else:
                print(f"Group {name}: {state.tolist()} (DOMs are {dom_experts[name]})")

        del grouper

        print("[TAMP] Number of parameters after merging:", model.num_parameters())
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # torch.save(model.state_dict(), output_path+"/model.pth")

    if eval_ppl:
        evaluate_minipile_perplexity(
            model, tokenizer=tokenizer, batch_size=eval_batch_size, log=True
        )
    elif isinstance(task, str):
        evaluate_fewshot(
            model, tokenizer=tokenizer, task=task, num_fewshot=num_fewshot, output_path=output_path, eval_batch_size=eval_batch_size, log=True
        )
    else:
        tasks = ["rte" , "arc_challenge", "arc_easy", "boolq", "hellaswag", "mmlu", "openbookqa", "winogrande"]
        eval_size = [32, 32, 32, 16, 32, 12, 32, 32]
        for i, t in enumerate(tasks):
            evaluate_fewshot(
                model, tokenizer=tokenizer, task=t, num_fewshot=num_fewshot, eval_batch_size=eval_size[i], output_path=result_path, log=True
            )


if __name__ == "__main__":
    Fire(evaluate_mcsmoe)
