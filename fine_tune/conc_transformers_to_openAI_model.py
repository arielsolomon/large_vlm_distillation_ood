
import torch
from transformers import CLIPModel, CLIPConfig
import open_clip

# Load your custom-trained transformers CLIP model
transformers_model_path = "/Data/federated_learning/large_vlm_distillation_ood/finetune_clip/clip-vit-large-patch14-finetuned-stanford-cars_state_dict_17_07.pt"
config = CLIPConfig.from_pretrained("openai/clip-vit-large-patch14")
transformers_clip = CLIPModel(config)
transformers_clip.load_state_dict(torch.load(transformers_model_path))

# Create a new OpenAI CLIP model instance
openai_clip = open_clip.create_model("ViT-L-14", pretrained=None)

# Mapping dictionary to rename keys from transformers format to openai format
key_mapping = {
    'text_model.encoder.layers': 'transformer.resblocks',
    'vision_model.encoder.layers': 'visual.transformer.resblocks',
    'text_projection': 'text_projection',
    'vision_projection': 'visual.proj',
    'logit_scale': 'logit_scale'
}

# Transform the state dict
transformers_state_dict = transformers_clip.state_dict()
openai_state_dict = {}

for key, value in transformers_state_dict.items():
    new_key = key
    for transformers_key, openai_key in key_mapping.items():
        if transformers_key in key:
            new_key = key.replace(transformers_key, openai_key)
            break
    openai_state_dict[new_key] = value

# Load the converted state dict into the openai model
openai_clip.load_state_dict(openai_state_dict, strict=False)

# Save the converted state dict
torch.save(openai_clip.state_dict(), "/Data/federated_learning/large_vlm_distillation_ood/finetune_clip/clip_fine_tuned_state_dict_openAI.pth")

print("Model conversion completed successfully.")
