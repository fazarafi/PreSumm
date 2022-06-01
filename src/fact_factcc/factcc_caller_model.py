# importable class from external package/usage
import wandb
import numpy as np
import torch

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer)
from pytorch_transformers import AdamW, WarmupLinearSchedule

from fact_factcc.factcc_utils import (convert_examples_to_features, output_modes, processors)

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


class FactccCaller():
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.args = None 
        self.initiate_model()

    def load_config(self):
        cfg_path = "configs/factcc_config.yaml"
        return yaml.safe_load(open(cfg_path, "r"))

    def make_model_input_single(self, args, batch, i):
        inputs = {'input_ids': torch.tensor([batch[0][i].tolist()], device=args["device"]),
                'attention_mask': torch.tensor([batch[1][i].tolist()], device=args["device"]),
                'token_type_ids': torch.tensor([batch[2][i].tolist()], device=args["device"]),
                'labels': torch.tensor([batch[3][i].tolist()], device=args["device"])}

        return inputs

    def load_model(checkpoint):
        wandb.init(project="entailment-metric")

    
    def initiate_model(self):
        cfg = self.load_config()

        self.args = self.parser(cfg)
        
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.args["model_type"]]
        
        self.tokenizer = tokenizer_class.from_pretrained(self.args["model_name_or_path"], do_lower_case=self.args["do_lower_case"])

        checkpoint = self.args["model_dir"]
        self.model = model_class.from_pretrained(checkpoint)
        self.model.to(self.args["device"])

    def classify(self, document, summary):

        result, result_output = self.evaluate(document, summary)
        # logger.info("[DEBUG FT] result: " + str(result))
        # logger.info("[DEBUG FT] result_output: " + str(result_output))
        return 1 if result_output[0]==0 else -1
        
    def evaluate(self, document, summary):
        # Loop to handle MNLI double evaluation (matched, mis-matched
        results = {}
        eval_dataset = self.convert_doc_sum_to_dataset(document, summary, evaluate=True)
        
        self.args["eval_batch_size"] = 1 * max(1, self.args["n_gpu"])
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if self.args["local_rank"] == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)

        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.args["device"]) for t in batch)

            with torch.no_grad():
                inputs = self.make_model_input_single(self.args, batch, 0)
                outputs = self.model(**inputs)
                logits_ix = 1 if self.args["model_type"] == "bert" else 7
                logits = outputs[logits_ix]
                
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        preds_output = np.argmax(preds, axis=1)
            
        return preds, preds_output

    def parser(self, args):
        if args["local_rank"] == -1 or args["no_cuda"]:
            device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
            args["n_gpu"] = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(args["local_rank"])
            device = torch.device("cuda", args["local_rank"])
            torch.distributed.init_process_group(backend='nccl')
            args["n_gpu"] = 1
        args["device"] = device

        return args

    def convert_doc_sum_to_dataset(self, document, summary, evaluate=False):
        if self.args["local_rank"] not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        task = self.args["task_name"]
        processor = processors[task]()
        output_mode = output_modes[task]
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(self.args["data_dir"], 'cached_{}_{}_{}_{}'.format(
            'dev' if evaluate else 'train',
            list(filter(None, self.args["model_name_or_path"].split('/'))).pop(),
            str(self.args["max_seq_length"]),
            str(task)))
        logger.info("Creating features from dataset file at %s", self.args["data_dir"])
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(document, summary)
        features = convert_examples_to_features(examples, label_list, self.args["max_seq_length"], self.tokenizer, output_mode,
            cls_token_at_end=bool(self.args["model_type"] in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=self.tokenizer.cls_token,
            cls_token_segment_id=2 if self.args["model_type"] in ['xlnet'] else 0,
            sep_token=self.tokenizer.sep_token,
            sep_token_extra=bool(self.args["model_type"] in ['roberta']),
            pad_on_left=bool(self.args["model_type"] in ['xlnet']),                 # pad on the left for xlnet
            pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
            pad_token_segment_id=4 if self.args["model_type"] in ['xlnet'] else 0)
        
        if self.args["local_rank"] == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_ext_mask = torch.tensor([f.extraction_mask for f in features], dtype=torch.float)
        all_ext_start_ids = torch.tensor([f.extraction_start_ids for f in features], dtype=torch.long)
        all_ext_end_ids = torch.tensor([f.extraction_end_ids for f in features], dtype=torch.long)
        all_aug_mask = torch.tensor([f.augmentation_mask for f in features], dtype=torch.float)
        all_aug_start_ids = torch.tensor([f.augmentation_start_ids for f in features], dtype=torch.long)
        all_aug_end_ids = torch.tensor([f.augmentation_end_ids for f in features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                all_ext_mask, all_ext_start_ids, all_ext_end_ids,
                                all_aug_mask, all_aug_start_ids, all_aug_end_ids)
        return dataset
