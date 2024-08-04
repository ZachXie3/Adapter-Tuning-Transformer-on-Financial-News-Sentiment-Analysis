import numpy as np
import random
import torch

import dataload
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from trainer import BertTrainer
from lora import add_lora_layers, freeze_model


if __name__ == "__main__":
    seed = 123456
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Hyperparameters
    model = "bert-base-uncased"  # "ProsusAI/finbert"
    num_epochs = 10
    batch_size = 64
    learning_rates = [1e-3, 1e-4, 1e-5]
    lora_ranks = [4, 8, 10]
    lora_alphas = [5, 10, 16]
    lora_dropouts = [0.05, 0.1, 0.2]

    train_loader, val_loader, test_loader = dataload.load_data(batch_size=batch_size, random_state=seed)

    for learning_rate in learning_rates:
        for lora_rank in lora_ranks:
            for lora_alpha in lora_alphas:
                for lora_dropout in lora_dropouts:
                    # load tokenizer and pretrained model
                    tokenizer_base = BertTokenizer.from_pretrained(model)
                    config = BertConfig.from_pretrained(model, num_labels=3)

                    # Load the BERT model for sequence classification with the specified configuration
                    bert_base = BertForSequenceClassification.from_pretrained(
                        model,
                        config=config
                    )

                    # #bert base
                    # trainer_bert_base = BertTrainer(
                    #     bert_base,
                    #     tokenizer_base,
                    #     lr=1e-04,
                    #     epochs=4,
                    #     train_dataloader=train_loader,
                    #     eval_dataloader=val_loader,
                    #     output_dir='./results/bert_base_fine_tuned',
                    #     output_filename='bert_base_lr1e-4',
                    #     save=True,
                    # )

                    # trainer_bert_base.train(evaluate=True)

                    add_lora_layers(
                        model=bert_base,
                        r=lora_rank,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout
                    )  # inject the LoRA layers into the model
                    freeze_model(bert_base)  # freeze the non-LoRA parameters

                    n_params = 0
                    n_trainable_params = 0

                    # count the number of trainable parameters
                    for n, p in bert_base.named_parameters():
                        n_params += p.numel()
                        if p.requires_grad:
                            n_trainable_params += p.numel()

                    print(f"Total parameters: {n_params}")
                    print(f"Trainable parameters: {n_trainable_params}")
                    percent_trainable = round(n_trainable_params / n_params * 100, 2)
                    print(f"Percentage trainable: {percent_trainable}%")

                    # bert base lora all r = 8
                    tuned_lora = BertTrainer(
                        model=bert_base,
                        tokenizer=tokenizer_base,
                        train_dataloader=train_loader,
                        eval_dataloader=val_loader,
                        test_dataloader=test_loader,
                        epochs=num_epochs,
                        lr=learning_rate,
                        output_dir='./results/bert_base_fine_tuned_lora_r10',
                        output_filename='bert_base_lora_r10_lr1e-4',
                        # save=True,
                    )
                    # tune lora on train-val dataset
                    tuned_lora.train(evaluate=True)

                    # plot learning curve
                    # metric options are ['accuracy', 'recall', 'precision', 'f1', 'loss']
                    for metric in ["accuracy", "recall", "precision", "f1", "loss"]:
                        tuned_lora.plot_learning_curve(
                            metric=metric,
                            r=lora_rank,
                            lr=learning_rate,
                            alpha=lora_alpha,
                            dropout=lora_dropout,
                            save=True)
                    # run tuned_lora on test dataset
                    tuned_lora.evaluate()

                    tuned_lora.save_data(n_params, n_trainable_params, percent_trainable, lora_rank, learning_rate,
                                         lora_alpha, lora_dropout)

