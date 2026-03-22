"""
As is explained in BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding by Devlin et al., the bi-directional model BERT is 
pre-trained by applying masked language modeling (MLM), where a percentage
of tokens of the input sequence are masked, and the objective of training is
to predict the vocabulary index of these these masked tokens. Here, we will
attempt to replicate this approach as we apply language transfer from multi-
lingual to Ancient Greek, following the steps laid out in BERT in Plutarch's 
Shadows by Yamshchikovm et al. 
"""
import os
import argparse
import torch
from torch.optim import AdamW
from torch.nn import functional as F
from torch.utils.data import (
    Dataset,
    DataLoader
)
import tqdm
import random
from transformers import BertTokenizer
from models.basic_bert import CustomBertModel
from transformers.utils import logging
logging.set_verbosity_error() 


def return_arguments(arguments: argparse.Namespace = None) -> argparse.Namespace:
    # We will define the basic arguments for the rest of the pipeline.
    arguments = argparse.Namespace() if arguments is None else arguments

    # General arguments:
    arguments.checkpoint = "google-bert/bert-base-multilingual-cased"

    # Dataset arguments:
    arguments.raw_text_dir = "data/text"
    arguments.processed_text_dir = "data/processed_text"
    arguments.max_sequence_length = 512
    arguments.batch_size = 8
    arguments.percent_masked = 15
    arguments.file_limit = 10

    # Training arguments
    arguments.epochs = 5
    arguments.lr = 3e-4
    arguments.gradient_clipping = True

    return arguments


def return_dataloader_for_mlm(arguments: argparse.Namespace = None) -> DataLoader:

    arguments = return_arguments() if arguments is None else arguments
    files = [
        f"{arguments.raw_text_dir}/{f}" 
        for f in os.listdir(arguments.raw_text_dir)
    ]
    
    class DatasetForMLM(Dataset):
        def __init__(self, arguments=arguments, files=files, file_limit=10):
            super().__init__()
            self.files = files if file_limit is None else files[:file_limit]
            self.tokenizer = BertTokenizer.from_pretrained(arguments.checkpoint)

            input_ids = []
            for file in tqdm.tqdm(self.files):
                with open(file, "r", errors="ignore") as f:
                    input_ids.extend(self.tokenizer.encode(f.read(), add_special_tokens=False))

            self.chunks = [
                input_ids[i: i+arguments.max_sequence_length] 
                for i in range(0, len(input_ids) - arguments.max_sequence_length + 1, arguments.max_sequence_length)
            ]

        def __len__(self): 
            return len(self.chunks)

        def __getitem__(self, idx):
            input_ids = torch.tensor(self.chunks[idx])

            # 15% of input tokens are masked
            # 80% of the time, we replace the masked input token with `[MASK]`
            # 10% of the time, we replace the masked input token with a random token
            # 10% of the time, keep the original token
            mlm_mask = torch.bernoulli(torch.ones_like(input_ids) * 0.15 * 0.1)
            mlm_replace_with_random = torch.bernoulli(torch.ones_like(input_ids) * 0.15 * 0.1)
            mlm_replace_with_mask = torch.bernoulli(torch.ones_like(input_ids) * 0.15 * 0.8)
            mlm_mask = mlm_mask.masked_fill(mlm_replace_with_random==1, 1).masked_fill(mlm_replace_with_mask==1, 1)

            corrupted_input_ids = input_ids.masked_fill(mlm_replace_with_random==1, random.randint(0, self.tokenizer.vocab_size-1))
            corrupted_input_ids = corrupted_input_ids.masked_fill(mlm_replace_with_mask==1, self.tokenizer.mask_token_id)

            return input_ids, corrupted_input_ids, mlm_mask
        
    def collate_fn(batch):
        input_ids = torch.stack([input_ids for input_ids, _, _ in batch])
        currupted_input_ids = torch.stack([currupted_input_ids for _, currupted_input_ids, _ in batch])
        mlm_mask = torch.stack([mlm_mask for _, _, mlm_mask in batch])
        return {
            "input_ids": input_ids, "corrupted_input_ids": currupted_input_ids, "mlm_mask": mlm_mask
        }

        
    dataset = DatasetForMLM(arguments=arguments, files=files, file_limit=arguments.file_limit)
    dataloader = DataLoader(dataset, batch_size=arguments.batch_size, shuffle=True, collate_fn=collate_fn)

    return dataloader


if __name__ == "__main__":

    arguments = return_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = return_dataloader_for_mlm(arguments)

    model = CustomBertModel.from_pretrained(arguments.checkpoint).to(device)
    optimizer = AdamW(model.parameters(), arguments.lr)
    loss_fn = F.cross_entropy

    history = {"training_loss": [0] * arguments.epochs}

    for epoch in range(arguments.epochs):
        model.train()
        for batch in tqdm.tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            corrupted_input_ids = batch["corrupted_input_ids"].to(device)
            mlm_mask = batch["mlm_mask"].to(device)

            attention_mask = torch.ones_like(input_ids).to(device)
            token_type_ids = torch.zeros_like(input_ids).to(device)

            last_hidden_state = model(corrupted_input_ids, token_type_ids, attention_mask)["last_hidden_state"]
            predicted_ids = model.hidden_state_to_token(last_hidden_state)
            
            loss = loss_fn(predicted_ids[mlm_mask==1], input_ids[mlm_mask==1])
            
            optimizer.zero_grad()
            loss.backward()

            if arguments.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            history["training_loss"][epoch] += loss.item()

        history["training_loss"][epoch] /= len(dataloader)
        print(f"Epoch: {epoch + 1} of {arguments.epochs}. Average loss: {history['training_loss'][epoch]}")

