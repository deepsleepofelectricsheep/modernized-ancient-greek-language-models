"""
Here, we will finetune BERT on the authorship classification dataset,
following the instructions laid out by Yamshikov et al.. 
"""
import os
import argparse
import torch 
from typing import List
from torch.optim import AdamW
from torch.nn import functional as F
from torch.optim.lr_scheduler import (
    LinearLR,
    SequentialLR
)
from torch.utils.data import (
    Dataset,
    DataLoader
)
import tqdm
from transformers import (
    BertTokenizer, 
    BertConfig
)
from models.basic_bert import CustomBertModel
from transformers.utils import logging
logging.set_verbosity_error() 


def return_arguments(arguments: argparse.Namespace = None) -> argparse.Namespace:
    # We will define the basic arguments for the rest of the pipeline.
    arguments = argparse.Namespace() if arguments is None else arguments

    # General arguments:
    arguments.checkpoint = "google-bert/bert-base-multilingual-cased"
    arguments.save_dir = "saved_models"
    arguments.saved_pretrained_model_fname = "ancient-greek-bert.pt"
    arguments.save_fname = "ancient-greek-bert-for-authorship-clf.pt"
    arguments.n_classes = 16

    # Dataset arguments:
    arguments.data_dir = "data/authorship_classification"
    arguments.max_sequence_length = 512
    arguments.batch_size = 32
    arguments.batch_limit = 10

    # Training arguments
    arguments.epochs = 2
    arguments.lr = 1e-4
    arguments.warmup_steps = 25
    arguments.decay_steps = 25
    arguments.gradient_clipping = True

    return arguments


def return_dataloaders_for_authorship_clf(
    arguments: argparse.Namespace = None
) -> List[DataLoader]:
    # Load and process train, dev and test data into dataloaders
    arguments = return_arguments() if arguments is None else arguments

    with open(f"{arguments.data_dir}/train.txt", "r") as f:
        train_data = f.readlines()
    with open(f"{arguments.data_dir}/dev.txt", "r") as f:
        dev_data = f.readlines()    
    with open(f"{arguments.data_dir}/test.txt", "r") as f:
        test_data = f.readlines()

    class DatasetForAuthorshipClf(Dataset):
        def __init__(self, data, arguments=arguments):
            super().__init__()
            self.sentences = [datum.split("\t")[1] for datum in data[1:]]
            self.author_idx = [datum.split("\t")[2] for datum in data[1:]]

            self.tokenizer = BertTokenizer.from_pretrained(arguments.checkpoint)

        def __len__(self):
            return len(self.sentences)
        
        def __getitem__(self, idx):
            tokenized_sentence = self.tokenizer.encode(self.sentences[idx], add_special_tokens=False)
            return torch.tensor(tokenized_sentence), int(self.author_idx[idx])
        
    def collate_fn(batch):
        # Pad to the maximum length 
        max_sequence_length = max(max([len(sentence) for sentence, author_idx in batch]), arguments.max_sequence_length)
        input_ids = torch.stack([F.pad(sentence, (0, max_sequence_length - len(sentence))) for sentence, author_idx in batch])
        author_idx = torch.tensor([author_idx for sentence, author_idx in batch])
        return {"input_ids": input_ids, "author_idx": author_idx}
    
      
    train_ds = DatasetForAuthorshipClf(train_data[:arguments.batch_limit*arguments.batch_size] if arguments.batch_limit else train_data)
    dev_ds = DatasetForAuthorshipClf(dev_data[:arguments.batch_limit*arguments.batch_size] if arguments.batch_limit else dev_data)
    test_ds = DatasetForAuthorshipClf(test_data[:arguments.batch_limit*arguments.batch_size] if arguments.batch_limit else test_data)

    train_dl = DataLoader(train_ds, batch_size=arguments.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dl = DataLoader(dev_ds, batch_size=arguments.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=arguments.batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dl, dev_dl, test_dl


def train(arguments: argparse.Namespace = None) -> None:
    
    arguments = return_arguments() if arguments is None else arguments 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load custom BERT model after language transfer via MLM
    checkpoint = torch.load(f"{arguments.save_dir}/{arguments.saved_pretrained_model_fname}", weights_only=True)
    model = CustomBertModel(BertConfig.from_pretrained(arguments.checkpoint))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.initialize_classification_head(n_classes=arguments.n_classes)
    model.to(device)

    train_dl, dev_dl, test_dl = return_dataloaders_for_authorship_clf(arguments)

    optimizer = AdamW(model.parameters(), lr=arguments.lr)
    warmup_scheduler = LinearLR(optimizer, start_factor=1/3, end_factor=1, total_iters=arguments.warmup_steps)
    decay_scheduler = LinearLR(optimizer, start_factor=1, end_factor=1/3, total_iters=arguments.decay_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[arguments.warmup_steps])

    loss_fn = F.cross_entropy

    history = {
        "training_loss_per_epoch": [0] * arguments.epochs,
        "dev_accuracy_per_epoch": [0] * arguments.epochs,
        "train_accuracy_per_epoch": [0] * arguments.epochs,
        "train_wallclock_time_per_epoch": [0] * arguments.epochs,
        "grad_norm_per_epoch": [0.0] * arguments.epochs,
    }

    print(f"Starting training loop...")
    for epoch in range(arguments.epochs):
        model.train()
        for batch in tqdm.tqdm(train_dl):
            input_ids = batch["input_ids"].to(device)
            author_idx = batch["author_idx"].to(device)

            attention_mask = torch.ones_like(input_ids).to(device)
            token_type_ids = torch.zeros_like(input_ids).to(device)

            hidden_state = model(input_ids, token_type_ids, attention_mask)["last_hidden_state"][:, 0, :]
            logits = model.head(hidden_state)
            predictions = torch.argmax(logits, dim=-1)

            loss = loss_fn(logits, author_idx)

            optimizer.zero_grad()
            loss.backward()

            if arguments.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Store per batch metrics, and track epoch metrics
            is_correct = (predictions == author_idx).sum().item()
            history["training_loss_per_epoch"][epoch] += loss.item()
            history["training_accuracy_per_epoch"][epoch] += is_correct

        train_datasize = len(train_dl) * arguments.batch_size
        history["training_loss_per_epoch"][epoch] /= train_datasize
        history["training_accuracy_per_epoch"][epoch] /= train_datasize
        print(f"Epoch: {epoch + 1} of {arguments.epochs}. Average loss: {history['training_loss_per_epoch'][epoch]}. Accuracy: {history['training_accuracy_per_epoch'][epoch]}")


if __name__ == "__main__":

    train()