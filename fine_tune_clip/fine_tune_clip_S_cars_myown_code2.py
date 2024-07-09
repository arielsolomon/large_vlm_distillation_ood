import argparse
import gc
from functools import partial
from pathlib import Path
import clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import transforms, datasets
from tqdm import tqdm
from PIL import Image

# Define the dataset loader function
def load_raw_data(root, validation_split=0.2, batch_size=64, num_workers=2):
    data_dir = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/resnet_classification_on_s_cars_dataset/s_cars_ood_for_finetune/'
    train_dir = data_dir+'train'
    test_dir = data_dir+'test'

    # Assuming train_dir and test_dir have subdirectories named after classes
    train_dataset = datasets.ImageFolder(train_dir, transform=None)
    test_dataset = datasets.ImageFolder(test_dir, transform=None)

    # Split train_dataset into train and validation sets
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(validation_split * num_train)

    train_sampler = SubsetRandomSampler(indices[split:])
    valid_sampler = SubsetRandomSampler(indices[:split])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    classes = train_dataset.classes
    return train_loader, validation_loader, test_loader, classes

# Define the dataset wrapper class
class ImageTitleDatasetWrapper(Dataset):
    def __init__(self, dataset, classes, preprocess, ood=False):
        self.dataset = dataset
        self.tokenized_title_list = [clip.tokenize(f"a photo of a {c}") for c in classes]
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.preprocess = preprocess
        self.ood = ood

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        image = self.transform(image) if self.ood else image
        image = self.preprocess(image)
        title = self.tokenized_title_list[label]
        return image, label, title

# Function to convert models to float32
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

# Function to convert models to mixed precision
def convert_models_to_mix(model):
    clip.model.convert_weights(model)

# Evaluation function
@torch.no_grad()
def evaluate(loader, model, device):
    model.eval()
    total = 0
    correct = 0
    text_inputs = torch.cat(loader.dataset.tokenized_title_list).to(device)
    for images, labels, _ in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        image_features = model.encode_image(images)
        text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        _, predicted = similarity.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy

# Function to freeze embeddings
def freeze_embed(model):
    freeze_list = ['positional_embedding', 'text_projection', 'logit_scale',
                   'visual.class_embedding',
                   'visual.positional_embedding', 'visual.proj', 'visual.conv1.weight',
                   'visual.ln_pre.weight', 'visual.ln_pre.bias']
    for name, param in model.named_parameters():
        if any(freeze_layer in name for freeze_layer in freeze_list):
            param.requires_grad = False

# Main function for training and evaluation
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

    # Load CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # Load Stanford Cars dataset
    train_loader, validation_loader, test_loader, classes = load_raw_data(
        args.data_path, args.validation_split, args.batch_size, args.num_workers)

    # Create dataset wrappers
    train_set_original = ImageTitleDatasetWrapper(train_loader.dataset, classes, preprocess)
    train_set_ood = ImageTitleDatasetWrapper(train_loader.dataset, classes, preprocess, ood=True)
    validation_set_original = ImageTitleDatasetWrapper(validation_loader.dataset, classes, preprocess)
    validation_set_ood = ImageTitleDatasetWrapper(validation_loader.dataset, classes, preprocess, ood=True)

    # Create data loaders
    train_loader = DataLoader(train_set_original, batch_size=args.batch_size, shuffle=True)
    train_loader_ood = DataLoader(train_set_ood, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set_original, batch_size=args.batch_size, shuffle=False)
    validation_loader_ood = DataLoader(validation_set_ood, batch_size=args.batch_size, shuffle=False)

    # Evaluate baseline accuracy
    model.eval()
    val_acc_baseline = evaluate(validation_loader, model, device)
    print(f'Baseline validation accuracy: {val_acc_baseline:.2f}%')
    val_acc_ood_baseline = evaluate(validation_loader_ood, model, device)
    print(f'Baseline validation accuracy OOD: {val_acc_ood_baseline:.2f}%')

    # Train on standard data and evaluate
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.num_epochs):
        train_epoch(device, train_loader, validation_loader, model, optimizer, epoch)

    # Fine-tune on OOD data and evaluate after each epoch
    for epoch in range(args.num_epochs):
        train_epoch(device, train_loader_ood, validation_loader_ood, model, optimizer, epoch)

    # Save the trained model
    save_path = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/resnet_classification_on_s_cars_dataset/s_cars_ood/fine_tune_mycode/clip_vit_base_stanford_cars_model.pth'
    torch.save(model, save_path)
    print(f'Saved fine-tuned CLIP model to: {save_path}')

def train_epoch(device, train_loader, validation_loader, model, optimizer, epoch):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels, _ in tqdm(train_loader):
        images = images.to(device)
        texts = labels.to(device)

        optimizer.zero_grad()
        freeze_embed(model)

        logits_per_image, logits_per_text = model(images, texts)
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
        loss = (criterion(logits_per_image, ground_truth) + criterion(logits_per_text, ground_truth)) / 2

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Evaluate on validation set after each epoch
    model.eval()
    val_acc = evaluate(validation_loader, model, device)
    print(f'Epoch {epoch + 1}, Validation Accuracy: {val_acc:.2f}%')

if __name__ == "__main__":
    data_dir = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/resnet_classification_on_s_cars_dataset/s_cars_ood_for_finetune/'
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on Stanford Cars dataset")
    parser.add_argument("--data-path", type=str, default=data_dir,
                        help="Path to Stanford Cars dataset")
    parser.add_argument("--save-dir", type=str, default='./saved_models',
                        help="Directory to save trained model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--validation-split", type=float, default=0.2,
                        help="Fraction of training data to be used for validation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of subprocesses to use for data loading")
    parser.add_argument("--use-cuda", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    main(args)
