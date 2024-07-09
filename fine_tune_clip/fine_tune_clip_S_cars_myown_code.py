import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# Define your data paths
data_dir = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/resnet_classification_on_s_cars_dataset/s_cars_ood_for_finetune/'


# Paths to the datasets
train_dir = data_dir+'train'
test_dir = data_dir+'test'
# Transformations
def scale_to_0_1(image):
  """
  Scales the image pixel values to a range of 0 to 1.
  Args:
      image: A PIL image.
  Returns:
      A PIL image with scaled pixel values.
  """
  # Assuming all pixel values are positive
  max_value = float(image.convert('L').getextrema()[1])
  return transforms.functional.compose(
      transforms.functional.to_tensor,
      transforms.functional.normalize)(image.convert('RGB') / max_value)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    scale_to_0_1,  # Custom scaling function
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    scale_to_0_1,  # Custom scaling function
    transforms.ToTensor()
])

# Load datasets
train_dataset = ImageFolder(train_dir, transform=train_transform)
test_dataset = ImageFolder(test_dir, transform=test_transform)

# DataLoader setup
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load CLIP model
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

# CLIP Processor
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_model.to(device)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(clip_model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
def train_clip(model, train_loader, optimizer, device, clip_processor):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc="Training"):
        inputs, _ = batch
        inputs = inputs.to(device)

        # Preprocess inputs using CLIPProcessor
        inputs_preprocessed = clip_processor(text=None, images=inputs, return_tensors="pt").to(device)

        outputs = model(**inputs_preprocessed)
        loss = outputs.loss
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / len(train_loader)

# Testing loop
def evaluate_clip(model, test_loader, device, clip_processor):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs, _ = batch
            inputs = inputs.to(device)

            # Preprocess inputs using CLIPProcessor
            inputs_preprocessed = clip_processor(text=None, images=inputs, return_tensors="pt").to(device)

            outputs = model(**inputs_preprocessed)
            loss = outputs.loss
            test_loss += loss.item()

    return test_loss / len(test_loader)

# Training and evaluation
epochs = 10  # Adjust as needed
for epoch in range(epochs):
    train_loss = train_clip(clip_model, train_loader, optimizer, device, clip_processor)
    test_loss = evaluate_clip(clip_model, test_loader, device, clip_processor)

    scheduler.step()

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Save the trained model if needed
torch.save(clip_model,'/home/user1/ariel/fed_learn/large_vlm_distillation_ood/resnet_classification_on_s_cars_dataset/s_cars_ood/fine_tune_mycode/clip_vit_base_stanford_cars_model.pth')

print('Fine-tuning complete. Model saved.')
