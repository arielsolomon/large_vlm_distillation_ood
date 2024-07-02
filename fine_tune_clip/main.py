import argparse
import os.path
import time
from pathlib import Path
import torch
from models.model_factory import get_model_hub_names, get_model
from dataloaders.data_loaders_factory import get_data_loaders
from multi_process_federated.federated_trainer import federated_train
from trainers.fine_tune_train import fine_tune_train, freeze_all_layers_but_last
from trainers.simple_trainer import evaluate_on_loaders


# Function to parse command-line arguments
def get_command_line_arguments(parser):
    """
    Parse command-line arguments.

    Parameters:
    parser (argparse.ArgumentParser): Argument parser object

    Returns:
    argparse.Namespace: Parsed arguments
    """
    parser.add_argument("--data-path", type=str, default=f"{str(Path.home())}/datasets/cifar",
                        help="dir path for datafolder")
    parser.add_argument("--num-clients", type=int, default="1", help="Number of clients in federation")
    parser.add_argument("--num-rounds", type=int, default="100",
                        help="Number of federated training rounds in federation")
    parser.add_argument("--batch-size", type=int, default="512", help="Number of images in train batch")
    parser.add_argument("--model-name", type=str, choices=get_model_hub_names(), default='resnet44',
                        help='client node or server node')
    parser.add_argument("--avg-orig", type=bool, default=True,
                        help='Use `Robust fine-tuning of zero-shot model (https://arxiv.org/abs/2109.01903)`.'
                             ' Average the fine tuned model and the pre trained model')
    parser.add_argument("--freeze-all-but-last", type=int, default=0,
                        help='Use `Fine-Tuning can Distort Pretrained Features and'
                             ' Underperform Out-of-Distribution (https://arxiv.org/abs/2202.10054)`.'
                             ' fine tune the classification head (last n linear layers in the case of'
                             ' simple mlp heads) only ')

    parser.add_argument("--load-from", type=str,
                        default=f'{str(Path.home())}/saved_models/cifar/resnet44/saved_at_Wed Oct 18 16:09:42 2023.pt',
                        # default='',
                        help='Load a pretrained model from given path. Train from scratch if string empty')
    parser.add_argument("--preform-pretrain", type=bool, default=False,
                        help='Train model in a federated manner before fine tuning')

    parser.add_argument("--use-cuda", type=bool, default=True,
                        help='Use GPU. Use cpu if not')

    parser.add_argument("--saved-models-path", type=str, default=f'{str(Path.home())}/saved_models/cifar/resnet44',
                        help='Train model in a federated manner before fine tuning')
    args = parser.parse_args()
    return args


# Main function
def main():
    """
    Main function for Private Federated Learning Flower.
    """
    parser = argparse.ArgumentParser(description="Private Federated Learning Flower")

    args = get_command_line_arguments(parser)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )

    # Initialize the neural network model
    net: torch.nn.Module = get_model(model_name=args.model_name, device=device)

    if args.load_from:
        # Load pretrained weights
        assert os.path.exists(
            args.load_from), f'Given path for pre-trained weights does not exist. Got {args.load_from}'
        net.load_state_dict(torch.load(args.load_from))

    # Load data loaders for training and testing
    train_loader, train_loader_ood, test_loader, test_loader_ood, _ = get_data_loaders(data_path=args.data_path,
                                                                                       batch_size=args.batch_size)

    if args.preform_pretrain:
        # Train model in a federated manner before fine tuning

        assert os.path.exists(args.saved_models_path), f'Create path for saved models. Got {args.saved_models_path}'

        federated_train(num_standard_clients=args.num_clients, net=net, test_loader=test_loader,
                        test_loader_ood=test_loader_ood, train_loader=train_loader, num_rounds=args.num_rounds)

        torch.save(net.state_dict(), f'{args.saved_models_path}/saved_at_{time.asctime()}.pt')
    else:
        assert args.load_from, f'Fine tune a never trained model?'

    # Evaluate the model on the provided loaders
    [base_accuracy_on_regular_data, base_accuracy_on_ood_data] = evaluate_on_loaders(net=net, loaders=[test_loader,
                                                                                                       test_loader_ood])

    if args.freeze_all_but_last:
        assert args.load_from or args.preform_pretrain, 'Freeze random weights???'
        assert args.freeze_all_but_last > 0, \
            f'Expected positive number of layers to freeze. Got {args.freeze_all_but_last}'

        freeze_all_layers_but_last(model=net, num_layers_to_freeze=args.freeze_all_but_last)

    # Fine-tune the model
    fine_tune_train(net=net,
                    train_loader_ood=train_loader_ood,
                    test_loader_ood=test_loader_ood,
                    test_loader=test_loader,
                    base_accuracy_on_regular_data=base_accuracy_on_regular_data,
                    base_accuracy_on_ood_data=base_accuracy_on_ood_data,
                    avg_orig=args.avg_orig)


# Entry point
if __name__ == "__main__":
    main()
