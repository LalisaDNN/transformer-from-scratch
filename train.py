import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from data import get_dataloader
from model import get_model

conf = {
    # model configuration
    'model_name': 'transformer_mt',
    'd_model': 512,
    'num_heads': 8,
    'num_encoder_blocks': 6,
    'num_decoder_blocks': 6,
    'max_len': 512,
    'dropout': 0.1,
    'activation': 'relu',
    'use_bias': False,

    # data configuration
    'dataset_name': 'ncduy/mt-en-vi',
    'src_lang': 'en',
    'tgt_lang': 'vi',
    'src_model': 'bert-base-uncased',
    'tgt_model': 'vinai/bartpho-word',
    'batch_size': 32,

    # training configuration
    'num_epochs': 10,
    'learning_rate': 1e-4,
    'max_grad_norm': 1.0,
    'running_dir': '/home/master/dnn/transformer/runs',
    'checkpoint_dir': None,
}

def train(conf):
    runs_dir = conf['running_dir']
    if conf['checkpoint_dir'] is not None:
        print(f'Loading checkpoint from {conf["checkpoint_dir"]}')
        if not os.path.exists(conf['checkpoint_dir']):
            raise FileNotFoundError(f'Checkpoint directory {conf["checkpoint_dir"]} does not exist.')
        # Load checkpoint logic here (if needed)
        checkpoint_dir = os.path.abspath(conf['checkpoint_dir'])
        save_model_dir = os.path.dirname(checkpoint_dir)
        runs_dir = os.path.dirname(save_model_dir)
        conf_dir = os.path.join(runs_dir, 'conf.json')
        if os.path.exists(conf_dir):
            with open(conf_dir, 'r') as f:
                conf.update(json.load(f))
    else:
        day = datetime.now().strftime('%Y_%m_%d')
        runs_dir = os.path.join(conf['running_dir'], day)
        if os.path.exists(runs_dir):
            os.remove(runs_dir)
        os.makedirs(runs_dir, exist_ok=True)
        json.dump(conf, open(os.path.join(runs_dir, 'conf.json'), 'w'), indent=4)
    
    save_model_dir = os.path.join(runs_dir, 'model')
    os.makedirs(save_model_dir, exist_ok=True)

    log_gir = os.path.join(runs_dir, 'log')
    os.makedirs(log_gir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    writer = SummaryWriter(log_dir=log_gir)

    # Get dataloader
    dataloader, tokenizer = get_dataloader(conf)
    src_vocab_size = tokenizer[conf['src_lang']].vocab_size
    tgt_vocab_size = tokenizer[conf['tgt_lang']].vocab_size

    # Get model
    model = get_model(conf, src_vocab_size, tgt_vocab_size)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer[conf['tgt_lang']].pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=conf['learning_rate'])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)

    start_epoch = 0
    global_step = 0
    if conf['checkpoint_dir'] is not None:
        checkpoint = torch.load(conf['checkpoint_dir'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    for epoch in range(start_epoch, conf['num_epochs']):
        print(f"************** Epoch {epoch + 1} ***************")

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            total_loss = 0.0
            num_batches = 0

            for batch in tqdm(dataloader[phase], desc=f"{phase.capitalize()} Phase", units="batch"):
                src_input_ids = batch['src_input_ids'].to(device)
                src_attention_mask = batch['src_attention_mask'].to(device)
                tgt_input_ids = batch['tgt_input_ids'].to(device)
                tgt_attention_mask = batch['tgt_attention_mask'].to(device)
                tgt_causal_mask = batch['tgt_causal_mask'].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(src_input_ids, tgt_input_ids, src_attention_mask, tgt_causal_mask)
                    loss = criterion(output.view(-1, output.size(-1)), tgt_input_ids.view(-1))

                    if phase == 'train':
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), conf['max_grad_norm'])
                        optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                global_step += 1
                writer.add_scalar(f'{phase}/loss_step', loss.item(), global_step)

            avg_loss = total_loss / num_batches
            writer.add_scalar(f'{phase}/loss_avg', avg_loss, epoch)

            if phase == 'train':
                lr_scheduler.step(avg_loss)

            print(f"{phase.capitalize()} Avg Loss: {avg_loss:.4f}")
        




