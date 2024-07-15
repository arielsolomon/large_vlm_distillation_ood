import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
from tqdm import tqdm

# Define paths
root = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/resnet18_classification_on_s_cars_dataset/s_cars_ood_ind/'
train_dir = root + 'train/'
# test_dir = root + 'train_eval_ind/'

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
# test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# Download the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Set up the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Custom contrastive loss function
def contrastive_loss(image_features, text_features, temperature=1.0):
    # Normalize features
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)

    # Calculate logits
    logits_per_image = image_features @ text_features.t() / temperature
    logits_per_text = logits_per_image.t()

    # Create ground-truth labels
    labels = torch.arange(len(logits_per_image)).to(device)

    # Calculate cross-entropy loss
    loss_img = F.cross_entropy(logits_per_image, labels)
    loss_txt = F.cross_entropy(logits_per_text, labels)

    return (loss_img + loss_txt) / 2

# Training loop
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        labels = labels.to(device)

        # Get text labels for the batch
        text_labels = [train_dataset.classes[label] for label in labels]

        # Preprocess the images and text with the processor, setting do_rescale=False
        image_inputs = processor(images=images, return_tensors="pt", do_rescale=False).to(device)
        text_inputs = processor(text=text_labels, return_tensors="pt", padding=True).to(device)

        # Extract image and text embeddings
        image_features = model.get_image_features(pixel_values=image_inputs['pixel_values'])
        text_features = model.get_text_features(input_ids=text_inputs['input_ids'], attention_mask=text_inputs['attention_mask'])
        # Compute contrastive loss
        loss = contrastive_loss(image_features, text_features)
        total_loss += loss.item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save the fine-tuned model
model.save_pretrained("clip-vit-large-patch14-finetuned-stanford-cars")
processor.save_pretrained("clip-vit-large-patch14-finetuned-stanford-cars")

print("Training complete and model saved.")
