import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

import torch
# import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    T5TokenizerFast,
    get_scheduler,
    SwitchTransformersForConditionalGeneration as HFSwitch,
)
from transformers.utils import logging as hf_logging

from mcsmoe.data import (
    Seq2SeqDataPreProcessor,
    tokenize_seq2seq,
    TASK_MAPPING_DATASET_ARGUMENTS,
    DataCollatorForSeq2Seq,
    get_evaluate_fn,
    EXTRA_KEYS_FOR_EVAL,
    keep_only_supporting_facts_in_context_for_hotpotqa
)
from mcsmoe.merging import (
    ExpertsGrouperForSwitch,
    merge_by_groups,
    merge_by_groups_within_and_across_models
)
from mcsmoe.models import (
    SwitchTransformersWrapperForDistillation
)

logger = get_logger(__name__)
logger.setLevel(20)
hf_logging.set_verbosity_warning()


def sanitize_merging_layers(layers: Union[str, List, int]):
    if layers is None:
        layers = list(range(1, 12, 2))
    elif isinstance(layers, str) and len(layers) > 0:
        layers = [int(x) for x in layers.split(",")]
    elif isinstance(layers, str) and len(layers) == 0:
        layers = []
    elif isinstance(layers, int):
        layers = [layers]
    return layers


@dataclass
class SwitchDistillationConfig:
    max_steps: Optional[int] = 10000
    adam_epsilon: Optional[float] = 1e-6
    adam_betas: Optional[Tuple[float, float]] = (0.9, 0.98)
    lr: Optional[float] = 5e-4
    warmup_steps: Optional[int] = 10
    weight_decay: Optional[float] = 0.01
    dropout: Optional[float] = 0.1
    mlm_probability: Optional[float] = 0.15
    mean_noise_span_length: Optional[int] = 3
    # === distillation parameters ===
    kd_temperature: Optional[float] = 2.0
    mlm_lambda: Optional[float] = 1.0
    kd_lambda: Optional[float] = 0.2
    reg_lambda: Optional[float] = 0.0
    hd_lambda: Optional[float] = 0.0
    hd_cos_sim: Optional[bool] = False
    next_rank_for_norm: Optional[int] = 384


def random_merge_and_distill_downstream_for_recover(
        output_dir: Optional[str] = None,
        teacher_checkpoint: Optional[str] = None,
        student_checkpoint: Optional[str] = None,
        task: Optional[str] = "sst2",
        similarity_base: Optional[str] = "router-logits",
        num_groups: Optional[int] = 8,
        strategy: Optional[str] = "average",
        encoder_merging_layers: Optional[Union[str, List, int]] = None,
        decoder_merging_layers: Optional[Union[str, List, int]] = None,
        # === training parameters ===
        no_eval_until_epochs: Optional[Union[int, float]] = 0,
        num_eval_steps: Optional[int] = None,
        log_steps: Optional[int] = 1,
        weight_decay: Optional[float] = 0.01,
        learning_rate: Optional[float] = 1e-3,
        gradient_accumulation_steps: Optional[int] = 32,
        warmup_steps: Optional[int] = 100,
        num_epochs: Optional[int] = 10,
        max_train_steps: Optional[int] = 20000,
        preprocessing_num_workers: Optional[int] = None,
        per_device_train_batch_size: Optional[int] = 8,
        per_device_eval_batch_size: Optional[int] = 8,
        # === distillation parameters ===
        kd_temperature: Optional[float] = 2.0,
        mlm_lambda: Optional[float] = 1.0,
        kd_lambda: Optional[float] = 0.2,
        hd_lambda: Optional[float] = 0.0,
):
    if output_dir is None:
        raise ValueError("output_dir must be specified")
    else:
        output_dir = os.path.join(output_dir, "random-merge")
        os.makedirs(output_dir, exist_ok=True)
    if teacher_checkpoint is None:
        raise ValueError("teacher_checkpoint must be specified")
    if student_checkpoint is None:
        student_checkpoint = teacher_checkpoint

    encoder_merging_layers = sanitize_merging_layers(encoder_merging_layers)
    decoder_merging_layers = sanitize_merging_layers(decoder_merging_layers)

    training_config = SwitchDistillationConfig(
        max_steps=max_train_steps,
        lr=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        mlm_probability=0.15,
        mean_noise_span_length=3,
        dropout=0.1,
        kd_temperature=kd_temperature,
        mlm_lambda=mlm_lambda,
        kd_lambda=kd_lambda,
        hd_lambda=hd_lambda,
        hd_cos_sim=False,
        reg_lambda=0
    )

    teacher_model = HFSwitch.from_pretrained(teacher_checkpoint)
    student_model = HFSwitch.from_pretrained(student_checkpoint)

    model = SwitchTransformersWrapperForDistillation(
        student=student_model,
        teacher=teacher_model,
        mlm_lambda=training_config.mlm_lambda,
        kd_lambda=training_config.kd_lambda,
        kd_temperature=training_config.kd_temperature,
        hd_lambda=training_config.hd_lambda,
        hd_cos_sim=training_config.hd_cos_sim,
        reg_lambda=training_config.reg_lambda,
    )

    tokenizer = T5TokenizerFast.from_pretrained("google/switch-base-8") # 現在發現 tokenizer 用錯會不會太遲
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           max_length=tokenizer.model_max_length,
                                           return_tensors='pt',
                                           keys_to_ignore=EXTRA_KEYS_FOR_EVAL)

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    set_seed(42)

    if accelerator.is_local_main_process:
        run_name = f"random-merge-{strategy}-{task}-{num_groups}"
        # wandb.init(project="mc-smoe",
        #            config={**student_model.config.__dict__,
        #                    **training_config.__dict__,
        #                    'num_params': student_model.num_parameters()},
        #            name=run_name)

    raw_dataset = load_dataset(*TASK_MAPPING_DATASET_ARGUMENTS[task])

    train_dataset = raw_dataset["train"]
    eval_dataset = raw_dataset["validation"] if task != "mnli" else (
        raw_dataset["validation_matched"], raw_dataset["validation_mismatched"]
    )

    with accelerator.main_process_first():
        if task == "hotpotqa":
            train_dataset = train_dataset.map(
                keep_only_supporting_facts_in_context_for_hotpotqa,
                batched=False,
                num_proc=preprocessing_num_workers
            )
            eval_dataset = eval_dataset.map(
                keep_only_supporting_facts_in_context_for_hotpotqa,
                batched=False,
                num_proc=preprocessing_num_workers
            )
        train_dataset = train_dataset.map(
            Seq2SeqDataPreProcessor(benchmark=task),
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=train_dataset.column_names
        )
        eval_dataset = eval_dataset.map(
            Seq2SeqDataPreProcessor(benchmark=task, keep_specific_keys=EXTRA_KEYS_FOR_EVAL),
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=eval_dataset.column_names
        )

    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenize_seq2seq(tokenizer=tokenizer, batch=x, keep_other_keys=False),
        num_proc=preprocessing_num_workers,
        batched=True,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False
    )
    tokenized_eval_dataset = eval_dataset.map(
        lambda x: tokenize_seq2seq(tokenizer=tokenizer, batch=x, keep_other_keys=True),
        num_proc=preprocessing_num_workers,
        batched=True,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False
    )

    logger.info(f"Number of training examples: {len(tokenized_train_dataset)}")
    logger.info(f"Number of validation examples: {len(tokenized_eval_dataset)}")
    dataset_for_merging = tokenized_train_dataset.select(range(128))

    train_dataloader = DataLoader(
        tokenized_train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=per_device_train_batch_size,
        num_workers=4
    )
    eval_dataloader = DataLoader(
        tokenized_eval_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=per_device_eval_batch_size,
        num_workers=4
    )
    merging_dataloader = DataLoader(
        dataset_for_merging,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=32,
        num_workers=4
    )

    print(f"Random merging")
    print(f"Number of groups: {num_groups}")
    print(f"Merge encoder layers: {encoder_merging_layers}")
    print(f"Merge decoder layers: {decoder_merging_layers}")

    grouper = ExpertsGrouperForSwitch(
        config=model.student.config,
        similarity_base=similarity_base,
    )
    grouper.compute_all_similarities(
        model=model.student,
        batch=next(iter(merging_dataloader)),
    )
    # grouper.compute_all_usages(
    #     model=model.student,
    #     batch=next(iter(merging_dataloader)),
    # )
    grouper.group_experts_by_knowledge(
        model=model.student,
        dataloader=merging_dataloader,
        num_groups=num_groups,
    )
    grouper.group_experts_into_clusters_by_routing_guided(
        num_groups=num_groups,
    )
    # grouper.group_experts_randomly(num_groups=num_groups)
    print(grouper.group_state_dict())
    torch.cuda.empty_cache()
    
    if strategy == "average":
        model.student = merge_by_groups(
            model=model.student,
            grouper=grouper,
            strategy="average",
            encoder_merging_layers=encoder_merging_layers,
            decoder_merging_layers=decoder_merging_layers,
            permute=False
        )
    elif strategy == "repair":
        model.student = merge_by_groups(
            model=model.student,
            grouper=grouper,
            strategy="average",
            encoder_merging_layers=encoder_merging_layers,
            decoder_merging_layers=decoder_merging_layers,
            permute=True,
            permute_strategy="activation-matching",
            dataloader=merging_dataloader,
        )
    elif strategy == "zipit":
        model.student = merge_by_groups_within_and_across_models(
            switch_model=model.student,
            grouper=grouper,
            dataloader=merging_dataloader,
            encoder_merging_layers=encoder_merging_layers,
            decoder_merging_layers=decoder_merging_layers,
            dominant_alone=False,
            usage_weighted=False
        )
    elif strategy == "git-rebasin":
        model.student = merge_by_groups(
            model=model.student,
            grouper=grouper,
            strategy="average",
            encoder_merging_layers=encoder_merging_layers,
            decoder_merging_layers=decoder_merging_layers,
            permute=True,
            permute_strategy="weight-matching",
        )

    print(f"Number of parameters before merging: {model.teacher.num_parameters()}")
    print(f"Number of parameters after merging: {model.student.num_parameters()}")

    no_weight_decay = ["bias", "LayerNorm", "mat_u", "mat_v"]  # no weight decay for orthogonalization matrices
    optimizer_grouped_params = [
        {
            "params": [
                p for n, p in model.student.named_parameters()
                if not any(nd in n for nd in no_weight_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.student.named_parameters()
                if any(nd in n for nd in no_weight_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_params,
                                  lr=learning_rate,
                                  eps=training_config.adam_epsilon,
                                  betas=training_config.adam_betas,
                                  weight_decay=weight_decay)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_epochs * num_update_steps_per_epoch
    no_eval_until_steps = no_eval_until_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps
    )
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    evaluate_fn = get_evaluate_fn(
        task=task,
        tokenizer=tokenizer,
        raw_eval_dataset=raw_dataset["validation"]
    )

    # ========================= Training ================================
    # Not using `accelerate_run_train` method here

    num_eval_steps = num_update_steps_per_epoch if num_eval_steps is None else num_eval_steps
    total_batch_size = (
            per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    ) if accelerator.num_processes is not None else (
            per_device_train_batch_size * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Num Samples = {len(train_dataloader)}")
    logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    best_eval = 0

    # Train!
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                kd_loss = outputs.kd_loss
                hd_loss = outputs.hd_loss
                task_loss = outputs.task_loss
                norm_loss = outputs.norm_loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_local_main_process and completed_steps % log_steps == 0 and (
                    accelerator.sync_gradients
            ):
                logger.info(f"epoch {epoch}, step {step}: loss {loss.item()}")
                # wandb.log({"train_loss": loss.item(),
                #            "train_kd_loss": kd_loss.item(),
                #            "train_hd_loss": hd_loss.item(),
                #            "train_task_loss": task_loss.item(),
                #            "train_norm_loss": norm_loss.item(),
                #            "epoch": completed_steps / num_update_steps_per_epoch,
                #            "learning_rate": lr_scheduler.get_lr()}, step=completed_steps)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
            if (completed_steps % num_eval_steps == 0 and completed_steps >= no_eval_until_steps and
                accelerator.sync_gradients) or (completed_steps <= 1 and step == 0 and no_eval_until_steps == 0
            ):
                model.student.eval()
                losses = []
                output_labels = []
                output_predictions = []
                output_ids = [] if task in ["squad", "copa", "multirc", "squad_v2", "hotpotqa"] else None
                for eval_step, eval_batch in enumerate(eval_dataloader):
                    extra_keys_eval_batch = {}
                    for key in list(eval_batch.keys()):
                        if key in EXTRA_KEYS_FOR_EVAL:
                            extra_keys_eval_batch[key] = eval_batch.pop(key)
                    with torch.no_grad():
                        outputs = model(**eval_batch)
                    eval_labels = accelerator.gather(eval_batch['labels'])
                    output_labels += torch.cat([
                        eval_labels,
                        torch.ones(eval_labels.shape[0], tokenizer.model_max_length - eval_labels.shape[1],
                                   dtype=eval_labels.dtype,
                                   device=eval_labels.device) * -100
                    ], dim=-1)
                    eval_logits = accelerator.gather(outputs.student_logits)
                    output_predictions += eval_logits.argmax(dim=-1).tolist()
                    if task in ["squad", "squad_v2", "hotpotqa"]:
                        output_ids += extra_keys_eval_batch["id"]
                    elif task == "copa" or task == "multirc":
                        output_ids += extra_keys_eval_batch["idx"]
                    losses.append(accelerator.gather_for_metrics(outputs["task_loss"]))
                losses = torch.cat(losses)
                eval_loss = torch.mean(losses)
                output_labels = torch.stack(output_labels, dim=0)
                eval_res = evaluate_fn(predictions=output_predictions, labels=output_labels, ids=output_ids)
                metric_key = list(eval_res.keys())[0]
                eval_res["task_loss"] = eval_loss.item()

                if eval_res[metric_key] > best_eval:
                    best_eval = eval_res[metric_key]
                    accelerator.wait_for_everyone()
                    # wandb.summary["best_" + metric_key] = best_eval
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(os.path.join(output_dir, "best"),
                                                    is_main_process=accelerator.is_local_main_process,
                                                    save_function=accelerator.save,
                                                    safe_serialization=False)
                    if accelerator.is_local_main_process:
                        tokenizer.save_pretrained(os.path.join(output_dir, "best"))
                        print(f"Best model saved with best evaluation {metric_key}: {eval_res[metric_key]}")

                if accelerator.is_local_main_process:
                    print(f"Step {completed_steps}: eval loss {eval_res['task_loss']}")
                    eval_res = {("eval_" + k): v for k, v in eval_res.items()}
                    # wandb.log(eval_res, step=completed_steps)
                # model.student.train()

            if completed_steps >= max_train_steps:
                break

    # Finish Training!
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(os.path.join(output_dir, "latest"),
                                    is_main_process=accelerator.is_local_main_process,
                                    save_function=accelerator.save,
                                    safe_serialization=False)

    if accelerator.is_local_main_process:
        tokenizer.save_pretrained(os.path.join(output_dir, "latest"))

    # if accelerator.is_local_main_process and wandb is not None:
    #     wandb.finish()


if __name__ == "__main__":
    Fire(random_merge_and_distill_downstream_for_recover)
