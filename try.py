# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("AIChenKai/TinyLlama-1.1B-Chat-v1.0-x2-MoE")
model = AutoModelForCausalLM.from_pretrained("AIChenKai/TinyLlama-1.1B-Chat-v1.0-x2-MoE")
print(model)
print(model.config)
