# -----------------------------
# referenced https://github.com/alexriggio/BERT-LoRA-TensorRT/tree/main
#
# -----------------------------
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import os
import time
import matplotlib.pyplot as plt
from copy import copy


class BertTrainer:
    """ A training and evaluation loop for PyTorch models with a BERT like architecture. """
    
    def __init__(
        self, 
        model,
        tokenizer,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        epochs=5,
        lr=1e-05,
        output_dir='results/',
        output_filename='model_state_dict.pt',
        save=False,
    ):
        """
        Args:
            model: torch.nn.Module: = A PyTorch model with a BERT like architecture,
            tokenizer: = A BERT tokenizer for tokenizing text input,
            train_dataloader: torch.utils.data.DataLoader = 
                A dataloader containing the training data with "text" and "label" keys,
            eval_dataloader: torch.utils.data.DataLoader = 
                A dataloader containing the evaluation data with "text" and "label" keys,
            epochs: int = An integer representing the number epochs to train,
            lr: float = A float representing the learning rate for the optimizer,
            output_dir: str = A string representing the directory path to save the model,
            output_filename: string = A string representing the name of the file to save in the output directory,
            save: bool = A boolean representing whether or not to save the model,
        """
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = copy(model).to(self.device)
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.save = save
        self.eval_loss = float('inf')  # tracks the lowest loss so as to only save the best model  
        self.epochs = epochs
        self.epoch_best_model = 0  # tracks which epoch the lowest loss is in so as to only save the best model
        self.results = {'train': {}, 'val': {}}
        metrics = ['accuracy', 'recall', 'precision', 'f1', 'loss']
        for key in metrics:
            self.results['train'][key] = []
            self.results['val'][key] = []
    
    def train(self, evaluate=False):
        """ Calls the batch iterator to train and optionally evaluate the model."""
        start_time = time.time()
        
        metrics = ['accuracy', 'recall', 'precision', 'f1', 'loss']
        for epoch in range(self.epochs):
            train_results_epoch = self.iteration(epoch, self.train_dataloader, train=True)            
            if evaluate and self.eval_dataloader is not None:
                test_results_epoch = self.iteration(epoch, self.eval_dataloader, train=False)
            for key in metrics:
                self.results['train'][key].append(train_results_epoch[key])
                self.results['val'][key].append(test_results_epoch[key])

        end_time = time.time()        
        execution_time = end_time - start_time
        minutes = int(execution_time // 60)
        seconds = int(execution_time % 60)
        print(f"Training Time: {minutes} min {seconds} sec.")

    def evaluate(self):
        """ Calls the batch iterator to evaluate the model."""
        start_time = time.time()
        
        epoch = 0
        self.iteration(epoch, self.test_dataloader, train=False)
        
        end_time = time.time()        
        execution_time = end_time - start_time
        minutes = int(execution_time // 60)
        seconds = int(execution_time % 60)
        print(f"Evaluate Time: {minutes} min {seconds} sec.")

    def iteration(self, epoch, data_loader, train=True):
        """ Iterates through one epoch of training or evaluation"""
        
        # initialize variables
        loss_accumulated = 0.
        correct_accumulated = 0
        samples_accumulated = 0
        preds_all = []
        labels_all = []
        
        # self.model.train() if train else self.model.eval()
        
        # progress bar
        mode = "train" if train else "eval"
        batch_iter = tqdm(
            enumerate(data_loader),
            desc=f"EP ({mode}) {epoch}",
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )
        
        # iterate through batches of the dataset
        for i, batch in batch_iter:
            texts, labels = batch

            labels = labels.to(self.device)

            # tokenize data
            input_sequence = self.tokenizer(
                texts,   # text
                padding=True, 
                truncation=True,
                max_length=512, 
                return_tensors='pt', 
            )

            # forward pass
            input_ids = input_sequence['input_ids'].to(self.device)
            attention_mask = input_sequence['attention_mask'].to(self.device)
            logits = self.model(input_ids, attention_mask=attention_mask, labels=labels).logits

            # calculate loss
            loss = self.loss_fn(logits, labels)
    
            # compute gradient and update weights
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # calculate the number of correct predictions
            preds = logits.argmax(dim=-1)
            correct = preds.eq(labels).sum().item()
            
            # accumulate batch metrics and outputs
            loss_accumulated += loss.item()
            correct_accumulated += correct
            samples_accumulated += len(labels)
            preds_all.append(preds.detach())
            labels_all.append(labels.detach())
        
        # concatenate all batch tensors into one tensor and move to cpu for compatibility with sklearn metrics
        preds_all = torch.cat(preds_all, dim=0).cpu()
        labels_all = torch.cat(labels_all, dim=0).cpu()
        
        # metrics
        accuracy = accuracy_score(labels_all, preds_all)
        precision = precision_score(labels_all, preds_all, average='weighted')
        recall = recall_score(labels_all, preds_all, average='macro')  # Weighted recall is the same as accuracy.
        f1 = f1_score(labels_all, preds_all, average='weighted')
        avg_loss_epoch = loss_accumulated / len(data_loader)

        # return results for plotting learning curve
        # results_epoch = {
        #     'accuracy': round(accuracy, 4),
        #     'recall': round(recall, 4),
        #     'precision': round(precision, 4),
        #     'f1': round(f1, 4),
        #     'loss': round(avg_loss_epoch, 4)
        # }
        results_epoch = {
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'loss': avg_loss_epoch
        }
        
        # print metrics to console
        # print(
        #     f"samples={samples_accumulated}, \
        #     correct={correct_accumulated}, \
        #     acc={round(accuracy, 4)}, \
        #     recall={round(recall, 4)}, \
        #     prec={round(precision, 4)}, \
        #     f1={round(f1, 4)}, \
        #     loss={round(avg_loss_epoch, 4)}"
        # )
        
        # save the model if the evaluation loss is lower than the previous best epoch 
        if self.save and not train and avg_loss_epoch < self.eval_loss:
            
            # create directory and filepaths
            dir_path = Path(self.output_dir)
            dir_path.mkdir(parents=True, exist_ok=True)
            file_path = dir_path / f"{self.output_filename}_epoch_{epoch}.pt"
            
            # delete previous best model from hard drive
            if epoch > 0:
                file_path_best_model = dir_path / f"{self.output_filename}_epoch_{self.epoch_best_model}.pt"
                os.remove(file_path_best_model)

            # save model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, file_path)
            
            # update the new best loss and epoch
            self.eval_loss = avg_loss_epoch
            self.epoch_best_model = epoch

        return results_epoch
    
    def plot_learning_curve(self, metric: str, r: int, lr: float, alpha: int, dropout: float, save: bool = False):
        scores_train = self.results['train'][metric]
        scores_val = self.results['val'][metric]
        epoch_ls = range(self.epochs)
        plt.plot(epoch_ls, scores_train, label='train')
        plt.plot(epoch_ls, scores_val, label='validation')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.title(f'Learning Curve - {metric.capitalize()}\n' + f"r={r}, lr={lr}, a={alpha}, dropout={dropout}")

        if save:
            plt.savefig(f"./images/rank_{r}_lr_{lr}_alpha_{alpha}_dropout_{dropout}_{metric}.png")
            plt.close()
        else:
            plt.show()

    def save_data(self, n_params: int, n_trainable_params: int, percent_trainable: float, r: int, lr: float, alpha: int,
                  dropout: float):
        self.results["n_params"] = n_params
        self.results["n_trainable_params"] = n_trainable_params
        self.results["percent_trainable"] = percent_trainable

        with open(f"./output/rank_{r}_lr_{lr}_alpha_{alpha}_dropout_{dropout}.pkl", "wb") as file:
            pickle.dump(self.results, file)
