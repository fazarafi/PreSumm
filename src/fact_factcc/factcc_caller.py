# importable class from external package/usage
import wandb
import numpy as np
import torch

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer)
from pytorch_transformers import AdamW, WarmupLinearSchedule


import argparse
import glob
import logging
import os
import random
import datetime as dt
import yaml

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def make_model_input_single(args, batch, i):
    inputs = {'input_ids':        torch.tensor([batch[0][i].tolist()], device=args.device),
              'attention_mask':   torch.tensor([batch[1][i].tolist()], device=args.device),
              'token_type_ids':   torch.tensor([batch[2][i].tolist()], device=args.device),
              'labels':           torch.tensor([batch[3][i].tolist()], device=args.device)}

    # add extraction and augmentation spans for PointerBert model
    if args.model_type == "pbert":
        inputs["ext_mask"] = batch[4]
        inputs["ext_start_labels"] = batch[5]
        inputs["ext_end_labels"] = batch[6]
        inputs["aug_mask"] = batch[7]
        inputs["aug_start_labels"] = batch[8]
        inputs["aug_end_labels"] = batch[9]
        inputs["loss_lambda"] = args.loss_lambda

    return inputs

def load_model(checkpoint):
    wandb.init(project="entailment-metric")

def load_config():
    cfg_path = "configs/factcc_config.yaml"
    return yaml.safe_load(open(cfg_path, "r"))

def classify(document, summary):
    logger.info("[FT DEBUG] doc: ", str(document))
    logger.info("[FT DEBUG] sum: ", str(summary))
    # TODO remove for dummy
    return 0.2
    
    cfg = load_config()
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

    # results = {}
    # if args.do_eval and args.local_rank in [-1, 0]:
    #     checkpoints = [args.output_dir]
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #     for checkpoint in checkpoints:
            
    #         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    
    global_step = ""
    checkpoint = cfg.model_dir
    model = model_class.from_pretrained(checkpoint)
    model.to(args.device)
    result, result_output = evaluate(cfg, model, tokenizer, prefix=global_step)
            
    return result
    
def evaluate(config, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (config.task_name,)
    
    results = {}
    for eval_task in zip(eval_task_names):
        eval_dataset = load_and_cache_examples(config, eval_task, tokenizer, evaluate=True)

        config.eval_batch_size = config.per_gpu_eval_batch_size * max(1, config.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if config.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=config.eval_batch_size)

        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(config.device) for t in batch)

            with torch.no_grad():
                inputs = make_model_input_single(config, batch, 0)
                outputs = model(**inputs)
                logits_ix = 1 if config.model_type == "bert" else 7
                logits = outputs[logits_ix]
                
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        preds_output = np.argmax(preds, axis=1)
        
    return preds, preds_output
