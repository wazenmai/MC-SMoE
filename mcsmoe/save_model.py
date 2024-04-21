from transformers import T5TokenizerFast, get_scheduler, SwitchTransformersForConditionalGeneration
checkpoint = "/home/wazenmai/Storage/cache/huggingface/hub/models--google--switch-base-32"
model = SwitchTransformersForConditionalGeneration.from_pretrained(checkpoint)