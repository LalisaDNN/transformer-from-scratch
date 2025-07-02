import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer

def get_tokenizer(src_model, tgt_model):
    tokenizer = {
        'src': AutoTokenizer.from_pretrained(src_model),
        'tgt': AutoTokenizer.from_pretrained(tgt_model)
    }
    return tokenizer

class MTDataset(Dataset):
    def __init__(self, dataset, src_lang, tgt_lang, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src_text = self.dataset[idx][self.src_lang]
        tgt_text = self.dataset[idx][self.tgt_lang]

        src_encodings = self.tokenizer['src'](src_text, truncation=True, padding='max_length', 
                                              max_length=self.max_length, return_tensors="pt")
        tgt_encodings = self.tokenizer['tgt'](tgt_text, truncation=True, padding='max_length', 
                                              max_length=self.max_length, return_tensors="pt")

        # Create causal mask for target (lower triangular matrix)
        tgt_seq_len = len(tgt_encodings['input_ids'])
        tgt_causal_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), dtype=torch.uint8))

        return {
            'src_input_ids': torch.tensor(src_encodings['input_ids']),
            'src_attention_mask': torch.tensor(src_encodings['attention_mask']),
            'tgt_input_ids': torch.tensor(tgt_encodings['input_ids']),
            'tgt_attention_mask': torch.tensor(tgt_encodings['attention_mask']),
            'tgt_causal_mask': tgt_causal_mask,
            'src_text': src_text,
            'tgt_text': tgt_text
        }
    
def get_dataloader(conf):
    dataset = load_dataset(conf['dataset_name'])
    tokenizer = get_tokenizer(conf['src_model'], conf['tgt_model'])

    mt_dataset = {phase: MTDataset(dataset[phase], conf['src_lang'], conf['tgt_lang'], tokenizer, conf['max_length']) 
                  for phase in ['train', 'validation', 'test']}

    dataloader = {phase: DataLoader(mt_dataset[phase], batch_size=conf['batch_size'], shuffle=(phase=='train')) 
                  for phase in ['train', 'validation', 'test']}

    return dataloader, tokenizer


