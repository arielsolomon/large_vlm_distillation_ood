import argparse
import gc
from functools import partial
from pathlib import Path
import clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import os

from fine_tune_train import save_iteration_results
import torchvision.models as models

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

    parser.add_argument("--batch-size", type=int, default="512", help="Number of images in train batch")

    parser.add_argument("--use-cuda", type=bool, default=True,
                        help='Use GPU. Use cpu if not')

    args = parser.parse_args()
    return args


class ImageTitleDatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset, list_txt: list[str], preprocess, ood=False):
        # Initialize image paths and corresponding texts
        self._dataset = dataset

        self.tokenized_title_dict = {c: clip.tokenize(f"a photo of a {c}") for c in list_txt}
        self.tokenized_title_list = [clip.tokenize(f"a photo of a {c}") for c in list_txt]
        self._transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.ColorJitter(contrast=0.5, brightness=1.0),
                                              transforms.ToPILImage()])
        # Load the model
        self._preprocess = preprocess
        self._ood = ood

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx, ood=False):
        image, label = self._dataset[idx]
        image = self._transform(image) if self._ood else image
        image = self._preprocess(image)
        title = self.tokenized_title_list[label]
        return image, label, title


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


def convert_models_to_mix(model):
    clip.model.convert_weights(model)


@torch.no_grad()
def evaluate(loader: DataLoader, model: torch.nn.Module, device: torch.device) -> float:
    total: int = 0
    correct: int = 0
    text_inputs = torch.cat(loader.dataset.tokenized_title_list).to(device)
    for image, label, _ in loader:
        total += len(label)
        image = image.to(device)
        label = label.to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity.max(1)

        correct += (indices == label).sum().item()

        image, label, image_features, text_features, similarity, values, indices = (
            image.cpu(), label.cpu(), image_features.cpu(), text_features.cpu(), similarity.cpu(), values.cpu(),
            indices.cpu())
        del image, label, image_features, text_features, similarity, values, indices

    accuracy: float = 100.0 * float(correct) / float(total)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return accuracy


def freeze_embed(model):
    freeze_list: list[str] = ['positional_embedding', 'text_projection', 'logit_scale',
                              'visual.class_embedding',
                              'visual.positional_embedding', 'visual.proj', 'visual.conv1.weight',
                              'visual.ln_pre.weight', 'visual.ln_pre.bias']
    for n, p in model.named_parameters():
        p.requires_grad = n not in freeze_list


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_dir = "/Data/federated_learning/large_vlm_distillation_ood/Resnet18_classification/car_data/car_data/"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # Load data loaders for training and testing
    #train_set, validation_set, test_set, classes = load_raw_data(root=args.data_path)
    train_dataset = ImageFolder(dataset_dir+'train/', transform=ToTensor())
    test_dataset = ImageFolder(dataset_dir+'test/', transform=ToTensor())
    train_classes = os.listdir(dataset_dir+'train/')
    test_classes = os.listdir(dataset_dir+'test/')
    random_seed = 42
    torch.manual_seed(random_seed);
    val_percent = 0.1
    val_size = int(val_percent * len(train_dataset))
    train_size = len(train_dataset) - val_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_set_original = ImageTitleDatasetWrapper(train_dataset, train_classes, preprocess)
    train_set_ood = ImageTitleDatasetWrapper(train_dataset, train_classes, preprocess, ood=True)
    validation_set_original = ImageTitleDatasetWrapper(test_dataset, test_classes, preprocess)
    validation_set_ood = ImageTitleDatasetWrapper(test_dataset, test_classes, preprocess, ood=True)
    # test_set = ImageTitleDatasetWrapper(test_set, classes, preprocess)
    train_loader = DataLoader(train_set_original, batch_size=64, shuffle=True)
    train_loader_ood = DataLoader(train_set_ood, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_set_original, batch_size=512, shuffle=False)
    validation_loader_ood = DataLoader(validation_set_ood, batch_size=512, shuffle=False)
    # test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

    # Evaluate model - Baseline Accuracy
    model.eval()
    val_acc_baseline = evaluate(validation_loader, model, device)
    print(f'Baseline validation accuracy {val_acc_baseline}')
    val_acc_ood_baseline = evaluate(validation_loader_ood, model, device)
    print(f'Baseline validation accuracy OOD {val_acc_ood_baseline}')

    accuracy_list_on_original, accuracy_list_on_ood = [val_acc_baseline], [val_acc_ood_baseline]

    # Train a single epoch on standard CIFAR10 data
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    train_epoch(device, loss_img, loss_txt, 1, optimizer,
                train_loader, {}, model, 0)

    # Evaluate again after a single epoch on standard CIFAR10 data
    model.eval()
    val_acc_baseline = evaluate(validation_loader, model, device)
    accuracy_list_on_original.append(val_acc_baseline)
    print('Validation after 1 epoch of fine tune regular cifar10')
    print(f'Validation accuracy {val_acc_baseline}')
    val_acc_ood_baseline = evaluate(validation_loader_ood, model, device)
    accuracy_list_on_ood.append(val_acc_ood_baseline)
    print(f'Validation accuracy OOD {val_acc_ood_baseline}')

    # Fine tune on OOD data and evaluate after each batch
    num_epochs = 3
    train_epoch_fn = partial(train_epoch, device, loss_img, loss_txt, num_epochs, optimizer
                             , train_loader_ood,
                             {'original': (validation_loader, accuracy_list_on_original),
                              'OOD': (validation_loader_ood, accuracy_list_on_ood)})

    print('Fine tuning on OOD data')
    for epoch in range(num_epochs):
        epoch_loss = train_epoch_fn(model, epoch)
        print('Epoch', epoch, 'loss', epoch_loss)


def train_epoch(device, loss_img, loss_txt, num_epochs, optimizer,
                train_loader, validation_loaders: dict[str, (DataLoader, list[float])], model, epoch):
    epoch_loss = 0.0
    pbar = tqdm(train_loader, total=len(train_loader))
    model.train()
    for batch in pbar:
        iteration_loss = train_iteration(batch, device, loss_img, loss_txt, model, optimizer)
        epoch_loss += iteration_loss
        model.eval()
        s = f"Epoch {epoch}/{num_epochs}"
        for loader_name in validation_loaders:
            validation_loader, acc_list = validation_loaders[loader_name]
            val_acc: float = evaluate(validation_loader, model, device)
            acc_list.append(val_acc)
            s += f" Validation Acc {loader_name}: {val_acc}"
        if 'original' in validation_loaders and 'OOD' in validation_loaders:
            save_iteration_results(accuracy_list_on_standard_data=validation_loaders['original'][1],
                                   accuracy_list_on_ood_data=validation_loaders['OOD'][1])
        pbar.set_description(s + f"Loss: {iteration_loss:.4f}")

    return epoch_loss


def train_iteration(batch, device, loss_img, loss_txt, model, optimizer):
    freeze_embed(model)
    optimizer.zero_grad()
    images, _, texts = batch
    images = images.to(device)
    texts = texts.squeeze()
    texts = texts.to(device)
    # Forward pass
    logits_per_image, logits_per_text = model(images, texts)
    # Compute loss
    ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
    total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
    # Backward pass
    total_loss.backward()

    if device == "cpu":
        optimizer.step()
    else:
        convert_models_to_fp32(model)
        optimizer.step()
        convert_models_to_mix(model)
    return float(total_loss)


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use CLIP for few shot")
    args = get_command_line_arguments(parser)
    main(args)
