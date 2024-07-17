from trainer import DPOTrainer
from datasets import load_dataset
from typing import Tuple, Optional
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, TrainingArguments

@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default=None, metadata={"help": ("The base model checkpoint")})
    max_length: Optional[int] = field(default=None, metadata={"help": ("The max length of the input")})
    max_prompt_length: Optional[int] = field(default=None, metadata={"help": ("The max length of the prompt")})
    precompute_ref_log_probs: Optional[bool] = field(default=False, metadata={"help": ("Whether to precompute the reference log probabilities")})
    scale: Optional[bool] = field(default=False, metadata={"help": ("Whether to scale the loss")})
    reduce: Optional[bool] = field(default=False, metadata={"help": ("Whether to reduce the loss")})
    train_data: Optional[str] = field(default=None, metadata={"help": ("The training data")})

def get_model(model_path) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_ref = AutoModelForCausalLM.from_pretrained(model_path, device_map='balanced')
    model_pol = AutoModelForCausalLM.from_pretrained(model_path, device_map='balanced')
    return model_ref, model_pol, tokenizer

def process_data(tokenizer, args):
    shuffled_data = load_dataset('json', data_files=args.train_data)["train"]
    remove_columns = ['reward_gap', 'context', 'scale', 'reduce']
    def process_function(row: dict) -> dict:
        prompt = [dict(role='system', content='')]
        for each in row['context']: 
            role = 'user' if each['role'] == 'human' else 'assistant'
            prompt.append(dict(role=role, content=each['text']))
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        if args.scale is True:
            return dict(prompt=prompt, chosen=row['chosen']['text'], rejected=row['rejected']['text'], gap_factor=row['scale'])
        elif args.reduce is True:
            return dict(prompt=prompt, chosen=row['chosen']['text'], rejected=row['rejected']['text'], gap_factor=row['reduce'])
        return dict(prompt=prompt, chosen=row['chosen']['text'], rejected=row['rejected']['text'])
    train_set = shuffled_data.map(process_function).remove_columns(remove_columns)
    return train_set

def train(model_args, training_args):
    model_ref, model_pol, tokenizer = get_model(model_args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    train_set = process_data(tokenizer, model_args)
    dpotrainer = DPOTrainer(
        model = model_pol,
        ref_model = model_ref,
        args = training_args,
        train_dataset = train_set,
        tokenizer = tokenizer,
        max_length = model_args.max_length,
        max_prompt_length = model_args.max_prompt_length,
        precompute_ref_log_probs = model_args.precompute_ref_log_probs,
        scale = model_args.scale,
        reduce = model_args.reduce
    )
    dpotrainer.train()

def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    # reduce and scale cannot both be true
    assert not (model_args.reduce and model_args.scale)
    train(model_args, training_args)

if __name__ == "__main__":
    main()