# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/18
import os
import gc
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
        dominant: Optional[str] = "frequency", # random, frequency, knowledge
        similarity_base: Optional[str] = "router-logits",
        mode: Optional[str] = "normal", 
        merge: Optional[str] = "zipit", # zipit, freq
        num_fewshot: Optional[int] = 0,
        n_sentences: Optional[int] = 32,
        train_batch_size: Optional[int] = 4,
        eval_batch_size: Optional[int] = 32,
        partition: Optional[int] = 1,
        output_path: Optional[str] = None,
):
    print(f"Merge model {model_name} with {num_average_groups} group, {dominant} dominant + {similarity_base} grouping + zipit {mode} merge, evaluate on {task}")

    eval_ppl = (task == "minipile")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = Qwen2MoeForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # dataloader_for_merging = get_minipile_dataloder(
    #     tokenizer=tokenizer,
    #     block_size=512,
    #     batch_size=1,
    #     subset_ratio=0.1,
    # )

    dataloader_for_merging = get_calib_dataloder(
        dataset="c4",
        tokenizer=tokenizer,
        max_block_size=2048,
        n_blocks_for_stat=n_sentences, # 128, reduce size to avoid OOM
        batch_size=train_batch_size,
        num_workers=4,
    )

    # MC-SMoE!
    print(f"[MC-SMoE] Merging into average {num_average_groups} groups...")

    grouper = ExpertsGrouperForQwen2MoE(config=model.config, similarity_base=similarity_base)
    grouper.compute_all_similarities(model, dataloader_for_merging)
    
    if dominant == "random":
        grouper.group_experts_randomly(num_groups=num_average_groups)
        # Freq-merge
        model = merge_by_groups_with_usage_weighted(
            model, grouper=grouper, merging_layers=list(range(0, model.config.num_hidden_layers))
        )
    elif dominant == "frequency":
        grouper.compute_all_usages(model, dataloader_for_merging)
        dom_experts = grouper.group_experts_globally_from_dominant_experts(
            num_average_groups=num_average_groups, merging_layers=list(range(0, model.config.num_hidden_layers))
        )
        if merge == "freq":
            model = merge_by_groups_with_usage_weighted(
                model, grouper=grouper, merging_layers=list(range(0, model.config.num_hidden_layers))
            )
        else:
            model = merge_by_groups_within_and_across_models(
                qwen_model=model,
                grouper=grouper,
                dataloader=dataloader_for_merging,
                mode=mode,
                dominant_alone=False,
                usage_weighted=False
            )
    elif dominant == "knowledge":
        model = grouper.all_in_one_knowledge_dominant(
            model=model, 
            dataloader=dataloader_for_merging, 
            mode=mode,
            num_groups=num_average_groups,
        )
        dom_experts = grouper.core_experts
    else:
        raise ValueError(f"Unknown dominant type: {dominant}")
    


    print(f"[MC-SMoE] ========= Grouping results ========= ")
    for name, state in grouper.group_state_dict().items():
        if dom_experts is None:
            print(f"Group {name}: {state.tolist()}")
        else:
            print(f"Group {name}: {state.tolist()} (DOMs are {dom_experts[name]})")

    del grouper

    print("[MC-SMoE] Number of parameters after merging:", model.num_parameters())
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.save(model.state_dict(), output_path+"/model.pth")

    if eval_ppl:
        evaluate_minipile_perplexity(
            model, tokenizer=tokenizer, batch_size=eval_batch_size, log=True
        )
    elif isinstance(task, str):
        evaluate_fewshot(
            model, tokenizer=tokenizer, task=task, num_fewshot=num_fewshot, output_path=output_path, log=True
        )
    else:
        for t in task:
            evaluate_fewshot(
                model, tokenizer=tokenizer, task=t, num_fewshot=num_fewshot, output_path=output_path+f"_{t}", eval_batch_size=eval_batch_size, log=True
            )
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    Fire(evaluate_mcsmoe)
