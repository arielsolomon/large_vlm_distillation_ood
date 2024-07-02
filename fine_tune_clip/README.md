# Federated Learning and Few-Shot Training

This repository contains code and utilities for federated learning using the Flower framework and implementing few-shot training for models that were initially trained on out-of-distribution (OOD) data.

## Overview

Federated learning is a decentralized machine learning approach where multiple clients collaboratively train a global model without sharing their raw data. This repository demonstrates a federated learning setup using the [Flower](https://flower.dev/) framework.

Additionally, the repository provides a few-shot training mechanism, aiming to fine-tune models that were previously trained on out-of-distribution data, enhancing their performance on specific tasks.

## Features

- Federated learning using Flower framework
- Few-shot training for models trained on out-of-distribution data
- Utilization of CIFAR-10 dataset
- Client and server nodes for federated training
## Usage

To use this repository for federated learning and few-shot training, follow these steps:

1. **Setup Environment:**

    Ensure you have [Poetry](https://python-poetry.org/) installed to manage the project dependencies.

    ```bash
    # Install project dependencies using Poetry
    poetry install
    ```

2. **Data Preparation:**

    Prepare the CIFAR-10 dataset by following the data loading instructions provided in the `dataloaders/cifar10_loader.py` file.

3. **Training:**

    - Run the federated training process using the `poetry run` command and `main.py` script. Adjust the parameters as needed for your specific use case.

    - Optionally, enable few-shot training by specifying the path to pre-trained models using the `--load-from` argument.

    ```bash
    # Run federated training
    poetry run python main.py --num-clients 5 --num-rounds 20 --batch-size 128
    ```

4. **Fine-Tuning:**

    - Fine-tune the models using the `fine_tune_train` function from `trainers/fine_tune_train.py`.

5. **Evaluation:**

    - Evaluate the trained and fine-tuned models using the `evaluate_on_loaders` function from `trainers/simple_trainer.py`.

## Directory Structure

- `dataloaders/`: Contains data loading utilities, including CIFAR-10 dataset loader.
- `models/`: Defines the neural network architecture (e.g., CNN model).
- `multi_process_federated/`: Includes federated learning-related components like federated trainer and node types.
- `trainers/`: Contains training and evaluation scripts, including fine-tuning and simple training mechanisms.


