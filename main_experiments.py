# Modified from https://github.com/bearpaw/pytorch-classification
from __future__ import print_function
import argparse
import os
import shutil
import time
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models
from custom_data_loader import CLIPImageDataset
import numpy as np
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import wandb

wandb.init(project="distilize_r50_s_cars_ood_ind_dataset_on_clip_lr_0.001")  # Replace with your project name

# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser()

# Datasets
parser.add_argument('-d', '--data', default='/home/user1/ariel/fed_learn/large_vlm_distillation_ood/s_cars_ood_ind/', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--repeat-epochs', default=1, type=int, metavar='N',
                    help='repeat training batch in the same epoch')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=32, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=32, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--skip-val', action='store_true', help='skip validation during training')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--onecyclelr', action='store_true', help='use onecyclelr')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='resnet50_dist_against_fine_tuned_30_07_testing', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default=False, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')

parser.add_argument('--trained-resnet', default='/home/user1/ariel/fed_learn/large_vlm_distillation_ood/resnet50_dist_against_fine_tuned_28_07/model_best.pth', type=str, metavar='PATH',
                    help='path to trained resnet for eval')
# Openset-specific
parser.add_argument('--fine-tune-clip', default='/home/user1/ariel/fed_learn/large_vlm_distillation_ood/fine_tune_clip/clip-vit-large-patch14-finetuned-stanford-cars_state_dict_openAI_format_17_07.pt')
parser.add_argument('--use-clip', default=True,action='store_true', help='whether to use CLIP model')
parser.add_argument('--clip-repo', type=str, default='custom', choices=['clip', 'open_clip', 'custom'], help='what model to use, native or fine tuned')
parser.add_argument('--clip-model', type=str, default='ViT-L/14')
parser.add_argument('--clip-dataset', type=str, default='openai', choices=['openai', 'laion400m_e31', 'laion400m_e32', 'laion2b_s32b_b82k'])
parser.add_argument('--clip-align-image-classification', type=int, default=1, help="whether to use L-cls (vision-language alignment loss for classification) (0/1)")
parser.add_argument('--chatgpt-raw-text-file', type=str, default='/home/user1/ariel/fed_learn/large_vlm_distillation_ood/datasets/s_cars_ind_ood/chatgpt.txt', help="file containing label descriptions generated by chatgpt")
parser.add_argument('--clip-align-proximal-text-num', type=int, default=256, help="If >0, specifies the k in L-vlalign")
parser.add_argument('--clip-align-text-only-proximal', action='store_true', help="If --clip-align-image-classification, whether to only keep L-vlalign and remove L-cls")
parser.add_argument('--clip-filter-out-wrong-alignment', action='store_true', help="If L-vlalign is used, whether to filter out images whose teacher visual features are misaligned with their language labels")
parser.add_argument('--clip-align-image-aux-caption', action='store_true', help="Whether to use auxiliary captions")
parser.add_argument('--clip-align-image-mse', default=True,action='store_true', help="Whether to use L-mse")
parser.add_argument('--clip-align-image-mse-unnorm', action='store_true', help="Whether to use L-mse (unnormalized version)")
parser.add_argument('--clip-align-image-contrastive', default=True,action='store_true', help="Whether to use L-im-cst")
parser.add_argument('--clip-align-image-contrastive-mode', type=str, default='bidirectional', choices=['single', 'bidirectional'])
parser.add_argument('--label-path', type=str, default='/home/user1/ariel/fed_learn/large_vlm_distillation_ood/data/StanfordCars/label2text.txt', help='Path to label2text.txt')
parser.add_argument('--temperature', type=float, default=0.07, help='Temperature')
parser.add_argument('--few-shot-num', type=int, default=0, help='Number of few-shot examples')
parser.add_argument('--few-shot-method', type=str, default='None', help='Few-shot mode, support retrieval or finetune')
parser.add_argument('--prompt-learner', action='store_true', help='Whether to use prompt learner (CoOp)')
parser.add_argument('--prompt-learner-nctx', type=int, default=8, help='Number of CoOp context tokens')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate',default=False, dest='evaluate', action='store_true',
                    help='Evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='Use pre-trained model')
parser.add_argument('--advprop', default=False, action='store_true',
                    help='Use advprop or not')
parser.add_argument('--use-adam', action='store_true', help='Whether to use Adam optimizer')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--clip-gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES for the CLIP model')

args = parser.parse_args()
args.clip_align_image_classification = bool(args.clip_align_image_classification)
print("clip_align_image_classification = ", args.clip_align_image_classification)
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
use_cuda = torch.cuda.is_available()
assert use_cuda, "CUDA is not available"

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    cuda_device = f"cuda:{args.gpu_id}"
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Build CLIP teacher model
    if args.use_clip:
        clip_device = f'cuda:{args.clip_gpu_id}'
        print("clip_device", clip_device)
        if args.clip_repo == 'clip':
            import clip
            clip_model, clip_preprocess_orig = clip.load(args.clip_model, device=clip_device)
            if 'ViT' in args.clip_model or args.clip_model in ['RN50']:
                clip_preprocess = transforms.Compose([
                    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
                    transforms.CenterCrop(size=(224, 224)),
                    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                ])
            elif args.clip_model == 'RN50x16':
                clip_preprocess = transforms.Compose([
                    transforms.Resize(size=384, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
                    transforms.CenterCrop(size=(384, 384)),
                    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                ])
            else:
                raise NotImplementedError()
        elif args.clip_repo == 'open_clip':
            import open_clip
            clip_model, _, clip_preprocess_orig = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_dataset)
            clip_preprocess = transforms.Compose([
                transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
                transforms.CenterCrop(size=(224, 224)),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        elif args.clip_repo == 'custom':
            # Load your custom model
            print('\nEntering custom mode\n')
            import clip
            clip_model, clip_preprocess_orig = clip.load(args.clip_model, device=clip_device)
            print('\nLoading state_dict')
            custom_state_dict = torch.load(args.fine_tune_clip, map_location=clip_device)
            print('\nLoading state dict into clip\n')
            clip_model.load_state_dict(custom_state_dict)
            print('\nModel loaded with state dict\n')
            if 'ViT' in args.clip_model or args.clip_model in ['RN50']:
                clip_preprocess = transforms.Compose([
                    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
                    transforms.CenterCrop(size=(224, 224)),
                    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                ])
            elif args.clip_model == 'RN50x16':
                clip_preprocess = transforms.Compose([
                    transforms.Resize(size=384, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
                    transforms.CenterCrop(size=(384, 384)),
                    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                ])
            else:
                raise NotImplementedError()

        else:
            raise NotImplementedError(f"Unknown CLIP repo: {args.clip_repo}")
        print("clip_preprocess", clip_preprocess)
        clip_model.to(clip_device).eval()
        clip_model = clip_model.to(torch.float32)
        # Prevent CLIP model from updating during training
        for m in clip_model.parameters():
            m.requires_grad = False
    else:
        clip_model = clip_preprocess = clip_device = None

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    val_on_train_dir = os.path.join(args.data, 'val_on_train')

    if not os.path.exists(val_on_train_dir):
        val_on_train_dir = None
    if args.advprop:
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    # Build transformations
    if 'vit' not in args.arch:
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                normalize,
            ])
        pre_train_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([
                transforms.Lambda(lambda img: (img * 255.0).to(torch.uint8)),
                transforms.RandAugment(2, 9),
                transforms.Lambda(lambda img: (img / 255.0).to(torch.float32)),
                transforms.RandomResizedCrop(224),
                normalize,
            ])
        pre_train_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor()
        ])

    # Build datasets
    if not args.evaluate:
        if args.few_shot_num > 0 and args.few_shot_method == 'finetune':
            from custom_data_loader import FSCLIPImageDataset
            train_dataset = FSCLIPImageDataset(traindir,
                                               fs_dir=valdir,
                                               fs_num=args.few_shot_num,
                                               transform=pre_train_transform,
                                               clip_model=clip_model,
                                               clip_preprocess=clip_preprocess,
                                               clip_device=clip_device,
                                               use_caption=args.clip_align_image_aux_caption)
            valdir = train_dataset.valdir
        else:
            train_dataset = CLIPImageDataset(traindir,
                                            pre_train_transform,
                                            clip_model=clip_model,
                                            clip_preprocess=clip_preprocess,
                                            clip_device=clip_device,
                                            use_caption=args.clip_align_image_aux_caption,)
    else:
        # Evaluation only
        train_dataset = CLIPImageDataset(traindir,
                                        pre_train_transform,
                                        clip_model=None,
                                        clip_preprocess=None,
                                        clip_device=None,
                                        use_caption=False)

    if args.few_shot_num > 0 and args.few_shot_method == 'finetune':
        from custom_data_loader import FewShotSampler
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch,
            num_workers=args.workers, pin_memory=True,
            sampler=FewShotSampler(train_dataset, args.train_batch))
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch, shuffle=True,
            num_workers=args.workers, pin_memory=True)

    val_transform = transforms.Compose([
            transforms.CenterCrop(224),
            normalize,
        ])
    val_dataset = CLIPImageDataset(valdir,
                                   transforms.Compose([
                                        transforms.Resize([256, 256]),
                                        transforms.ToTensor()
                                    ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_on_train_dataset = None
    val_on_train_loader = None
    if val_on_train_dir is not None:
        val_on_train_dataset = CLIPImageDataset(val_on_train_dir,
                                                transforms.Compose([
                                                    transforms.Resize([256, 256]),
                                                    transforms.ToTensor()
                                                ]))
        val_on_train_loader = torch.utils.data.DataLoader(
            val_on_train_dataset,
            batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # Create student model
    extra_args = dict()
    if args.clip_model == 'ViT-B/32':
        clip_feats_dim = 512
    elif args.clip_model in ['RN50']:
        clip_feats_dim = 1024
    else:
        clip_feats_dim = 768
    if args.use_clip:
        extra_args['fc_out_dim'] = clip_feats_dim # match clip feature dimension
    if ('efficientnet' in args.arch or 'vit' in args.arch) and 'fc_out_dim' in extra_args:
        extra_args['num_classes'] = extra_args['fc_out_dim'] # we hack "num_classes" as if it refers to feature dim, since we are doing open-set learning
        extra_args.pop('fc_out_dim')
        
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        if 'efficientnet' in args.arch:
            extra_args['advprop'] = args.advprop
            # model = EfficientNet.from_pretrained(args.arch, advprop=args.advprop, **extra_args)
        model = models.__dict__[args.arch](pretrained=True, **extra_args)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](**extra_args)
        
    model = model.to(cuda_device)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        
    # If using CLIP as teacher, process language descriptions and build prompt learner (if specified)
    prompt_learner = text_encoder = None
    gen_text_fxn = None
    if args.use_clip:
        label2text = {}
        chatgpt_label2text = {}
        chatgpt_lines = []
        if args.chatgpt_raw_text_file is not None:
            with open(args.chatgpt_raw_text_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 0:
                        chatgpt_lines.append(line)
        with open(args.label_path, 'r') as f:
            idx = 0
            for line in f:
                line = line.strip().split(' ') # [class_name_in_dataset, id, natural_language_class_name]
                if len(line) > 0:
                    line[2] = line[2].replace('_', ' ')
                    label2text[line[0]] = line[2]
                    if args.chatgpt_raw_text_file is not None:
                        chatgpt_label2text[line[0]] = chatgpt_lines[idx]
                    idx += 1
        if args.chatgpt_raw_text_file is not None:
            assert len(list(label2text.keys())) == len(chatgpt_lines), f"{len(label2text.keys())} != {len(chatgpt_lines)}"
        
        # print("train class_to_idx", train_dataset.class_to_idx)
        # print("test class_to_idx", val_dataset.class_to_idx)
        
        if args.chatgpt_raw_text_file is not None:
            gen_text_fxn = lambda x: label2text[x] + " . " + chatgpt_label2text[x]
        else:
            gen_text_fxn = lambda x: label2text[x]
        
        train_text_labels = ["a photo of " + gen_text_fxn(x) for x in train_dataset.class_to_idx.keys()]
        val_text_labels = ["a photo of " + gen_text_fxn(x) for x in val_dataset.class_to_idx.keys()]
        if val_on_train_dataset is not None:
            val_on_train_text_labels = ["a photo of " + gen_text_fxn(x) for x in val_on_train_dataset.class_to_idx.keys()]
        else:
            val_on_train_text_labels = None
            
        if args.prompt_learner:
            from models.misc.prompt_learner import PromptLearner, TextEncoder
            prompt_learner = PromptLearner(clip_model, [gen_text_fxn(x) for x in train_dataset.class_to_idx.keys()], \
                                            [gen_text_fxn(x) for x in val_dataset.class_to_idx.keys()], 
                                            [gen_text_fxn(x) for x in val_on_train_dataset.class_to_idx.keys()] if val_on_train_dataset is not None else None,
                                            device=cuda_device,
                                            cocoop=False,
                                            n_ctx=args.prompt_learner_nctx,)
            text_encoder = TextEncoder(clip_model)
        else:
            prompt_learner = text_encoder = None
            

            
        if args.clip_repo == 'clip' or args.clip_repo=='custom': 
            train_text_features = clip_model.encode_text(clip.tokenize(train_text_labels, truncate=True).to(clip_device)).float().detach()
            val_text_features = clip_model.encode_text(clip.tokenize(val_text_labels, truncate=True).to(clip_device)).float().detach()
            if val_on_train_text_labels is not None:
                val_on_train_text_features = clip_model.encode_text(clip.tokenize(val_on_train_text_labels, truncate=True).to(clip_device)).float().detach()
            else:
                val_on_train_text_features = None
        elif args.clip_repo == 'open_clip':
            tokenize = open_clip.tokenizer.tokenize
            train_text_features = clip_model.encode_text(tokenize(train_text_labels).to(clip_device)).float().detach()
            val_text_features = clip_model.encode_text(tokenize(val_text_labels).to(clip_device)).float().detach()
            if val_on_train_text_labels is not None:
                val_on_train_text_features = clip_model.encode_text(tokenize(val_on_train_text_labels).to(clip_device)).float().detach()
            else:
                val_on_train_text_features = None
    else:
        train_text_features = val_text_features = val_on_train_text_features = None
        
    # Retrieval-based few-shot learning, if specified
    if args.few_shot_num > 0 and args.few_shot_method == 'retrieval':
        # construct few shot samples
        few_shot_features = {}
        support_set_idx = []
        assert args.use_clip
        
        # follows TIP-Adapter
        num_augs = 10
        aug_transform = transforms.Compose([
            transforms.RandAugment(2, 9),
        ])
        tmp_dataset = datasets.ImageFolder(valdir)
        for idx, (img, target) in enumerate(tmp_dataset):
            if target not in few_shot_features.keys():
                few_shot_features[target] = []
            if len(few_shot_features[target]) < args.few_shot_num:
                support_set_idx.append(idx)
                imgs = [img]
                for _ in range(num_augs - 1):
                    imgs.append(aug_transform(img))
                few_shot_features[target].extend(imgs)
                
        for k in few_shot_features.keys():
            inp = torch.tensor(np.stack([clip_preprocess_orig(x) for x in few_shot_features[k]])).to(clip_device)
            with torch.no_grad():
                out = clip_model.encode_image(inp).float()
            if cuda_device != clip_device:
                out = out.to(cuda_device)
            few_shot_features[k] = F.normalize(out, dim=-1)
        few_shot_features = torch.stack(list(few_shot_features.values())) # (num_classes, few_shot_num, dim)
        support_set_idx = torch.tensor(support_set_idx)
    else:
        few_shot_features = None
        support_set_idx = None

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    
    params_to_optimize = list(model.parameters())
    if args.use_clip and prompt_learner is not None:
        for name, param in prompt_learner.named_parameters():
            if param.requires_grad:
                params_to_optimize.append(param)
                # print("Prompt learner param to optimize:", name)
    if not args.use_adam:
        optimizer = optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=0.01) # hardcoded
    scheduler = None
    if args.onecyclelr:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=1, epochs=args.epochs)

    # Resume
    title = 'MainExp-' + args.arch
    if args.resume:
        # Load checkpoint.
        print(f'==> Resuming from checkpoint.. {args.resume}')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume, map_location=cuda_device)
        best_acc = checkpoint['best_acc']
        if args.few_shot_method != 'finetune':
            start_epoch = checkpoint['epoch']
        else:
            start_epoch = 0
        
        pth1 = Path(os.path.join(args.resume, 'checkpoint.pth')).parent.absolute()
        pth2 = Path(os.path.join(args.checkpoint, 'log.txt')).parent.absolute()
        if pth1 == pth2:
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=False)
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Val On Train Acc.'])
        
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
        if len(missing_keys) > 0:
            print("Missing model keys:", missing_keys)
        if len(unexpected_keys) > 0:
            print("Unexpected model keys:", unexpected_keys)
        if args.few_shot_method != 'finetune':
            optimizer.load_state_dict(checkpoint['optimizer'])
        if args.onecyclelr and args.few_shot_method != 'finetune':
            scheduler.load_state_dict(checkpoint['scheduler'])
            
        if prompt_learner is not None:
            if 'prompt_learner' in checkpoint.keys():
                prompt_learner_state_dict = checkpoint['prompt_learner']
                keys = list(prompt_learner_state_dict.keys())
                for k in keys:
                    if 'token_prefix' in k or 'token_suffix' in k:
                        prompt_learner_state_dict.pop(k)
                prompt_learner.load_state_dict(checkpoint['prompt_learner'], strict=False)
            else:
                print("No prompt learner in checkpoint, initializing prompt learner from scratch...")
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Val On Train Acc.'])
    
    # Logger average for the last 5 epochs
    avg_logger = Logger(os.path.join(args.checkpoint, 'log_avg.txt'), title=title)
    avg_logger.set_names(['Avg Train Loss.', 'Avg Val Loss.', 'Avg Train Acc.', 'Avg Val Acc.', 'Avg Val On Train Acc.'])
    avg_train_loss = AverageMeter()
    avg_val_loss = AverageMeter()
    avg_train_acc = AverageMeter() 
    avg_val_acc = AverageMeter()
    avg_val_on_train_acc = AverageMeter()

    if args.evaluate:
        model.state_dict(torch.load(args.trained_resnet))
        eval_log_path = os.path.join(args.checkpoint, 'log_eval.txt')
        resume_eval_log = os.path.exists(eval_log_path)
        eval_logger = Logger(eval_log_path, title=title, resume=resume_eval_log)
        eval_logger.set_names(['val macc', 'val on train macc'])
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda, val_transform, 
                                   val_text_features, clip_model, clip_preprocess, clip_device,
                                   few_shot_features=few_shot_features, support_set_idx=support_set_idx,
                                   prompt_learner=prompt_learner, text_encoder=text_encoder, prompt_mode='test')
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc), flush=True)
        if val_on_train_loader is not None:
            val_on_train_loss, val_on_train_acc = test(val_on_train_loader, model, criterion, start_epoch, use_cuda, val_transform,
                                        val_on_train_text_features, clip_model, clip_preprocess, clip_device,
                                        few_shot_features=few_shot_features, support_set_idx=support_set_idx,
                                        prompt_learner=prompt_learner, text_encoder=text_encoder, prompt_mode='val_on_train')
            print(' Val on Train Loss:  %.8f, Val on Train Acc:  %.2f' % (val_on_train_loss, val_on_train_acc), flush=True)
        eval_logger.append([test_acc, val_on_train_acc])
        eval_logger.close()
        return

    # Train and validation
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, scheduler)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))

        for _ in range(args.repeat_epochs):
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args.epochs, use_cuda, train_transform, 
                                        train_text_features, clip_model, clip_preprocess, clip_device,
                                        prompt_learner=prompt_learner,
                                        text_encoder=text_encoder,
                                        prompt_mode='train',
                                        clip_align_image_classification=args.clip_align_image_classification,
                                        clip_align_image_aux_caption=args.clip_align_image_aux_caption,
                                        clip_align_proximal_text_num=args.clip_align_proximal_text_num,
                                        clip_align_text_only_proximal=args.clip_align_text_only_proximal,
                                        clip_filter_out_wrong_alignment=args.clip_filter_out_wrong_alignment,
                                        clip_align_image_mse=args.clip_align_image_mse,
                                        clip_align_image_mse_unnorm=args.clip_align_image_mse_unnorm,
                                        clip_align_image_contrastive=args.clip_align_image_contrastive,
                                        clip_align_image_contrastive_mode=args.clip_align_image_contrastive_mode)
        if not args.skip_val:
            test_target_remap = None
            if clip_model is None:
                # For closed-set classification, remap in- and out-of-distribution label ids
                train_n_cls = len(train_dataset.class_to_idx.keys())
                val_n_cls = len(val_dataset.class_to_idx.keys())
                if args.few_shot_num > 0:
                    test_target_remap = [train_n_cls - val_n_cls, train_n_cls]
                else:
                    test_target_remap = [train_n_cls, train_n_cls + val_n_cls]
            test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda, val_transform, 
                                        val_text_features, clip_model, clip_preprocess, clip_device,
                                        target_remap=test_target_remap,
                                        few_shot_features=few_shot_features, support_set_idx=support_set_idx,
                                        prompt_learner=prompt_learner, text_encoder=text_encoder, prompt_mode='test')
            val_on_train_acc = 0.0
            if (args.epochs - epoch <= 5) and (val_on_train_loader is not None):
                # accuracy on unseen samples whose labels belong to the training set
                _, val_on_train_acc = test(val_on_train_loader, model, criterion, epoch, use_cuda, val_transform,
                                        val_on_train_text_features, clip_model, clip_preprocess, clip_device,
                                        prompt_learner=prompt_learner, text_encoder=text_encoder, prompt_mode='val_on_train')
            # append logger file
            logger.append([optimizer.param_groups[0]['lr'], train_loss, test_loss, train_acc, test_acc, val_on_train_acc])
            
            if args.epochs - epoch <= 5:
                avg_train_loss.update(train_loss)
                avg_val_loss.update(test_loss)
                avg_train_acc.update(train_acc)
                avg_val_acc.update(test_acc)
                avg_val_on_train_acc.update(val_on_train_acc)
                if args.epochs - epoch == 1:
                    avg_logger.append([avg_train_loss.avg, avg_val_loss.avg, avg_train_acc.avg, avg_val_acc.avg, avg_val_on_train_acc.avg])
                    avg_logger.close()       

            # save model
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            save_dict = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict() if scheduler is not None else None,
                }
            if prompt_learner is not None:
                save_dict['prompt_learner'] = prompt_learner.state_dict()
            save_checkpoint(save_dict, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)




def train(train_loader, model, criterion, optimizer, epoch, total_epochs, use_cuda,
          train_transform,
          train_text_features=None, clip_model=None, clip_preprocess=None, clip_device=None,
          prompt_learner=None,
          text_encoder=None,
          prompt_mode='train',
          clip_align_image_classification=False,
          clip_align_image_aux_caption=False,
          clip_align_proximal_text_num=-1,
          clip_align_text_only_proximal=False,
          clip_filter_out_wrong_alignment=False,
          clip_align_image_mse=False, clip_align_image_mse_unnorm=False,
          clip_align_image_contrastive=False,
          clip_align_image_contrastive_mode='single'):
    #initiate train_text_features:


    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_losses = AverageMeter()
    losses = AverageMeter()
    aux_mse_losses = AverageMeter()
    aux_contrastive_losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))

    avg_accuracy_per_class = None

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if clip_model is not None:
            inputs, outputs_clip = inputs

        if clip_align_image_aux_caption:
            outputs_clip, aux_clip_caption = outputs_clip
        else:
            aux_clip_caption = None

        inputs = train_transform(inputs)

        if use_cuda:
            inputs, targets = inputs.to(cuda_device), targets.to(cuda_device)
        if clip_model is not None:
            outputs_clip = outputs_clip.to(cuda_device)
        if aux_clip_caption is not None:
            aux_clip_caption = aux_clip_caption.to(cuda_device)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = None
        aux_mse_loss = None
        aux_contrastive_loss = None
        if clip_model is not None:
            outputs_norm = nn.functional.normalize(outputs, dim=-1)
            outputs_clip_norm = nn.functional.normalize(outputs_clip, dim=-1)
            if aux_clip_caption is not None:
                aux_clip_caption_norm = nn.functional.normalize(aux_clip_caption, dim=-1)

            ## input: normalized image features from student; output: unnormalized clip text features
            if prompt_learner is not None and text_encoder is not None:
                assert prompt_mode == 'train'
                prompts = prompt_learner(outputs_norm)
                train_text_features = [] # override the input train_text_features since the prompt is adaptive
                for pts_i in prompts: # (n_cls, n_tkn, ctx_dim)
                    minib = 128
                    tokenized_idx = 0
                    cur_train_text_feature = []
                    while tokenized_idx < len(prompt_learner.train_tokenized_prompts):
                        cur_train_text_feature.append(
                            text_encoder(pts_i[tokenized_idx:tokenized_idx+minib], prompt_learner.train_tokenized_prompts[tokenized_idx:tokenized_idx+minib])
                        )
                        tokenized_idx += minib
                    cur_train_text_feature = torch.cat(cur_train_text_feature, dim=0)
                    train_text_features.append(cur_train_text_feature[None, ...])
                train_text_features = torch.cat(train_text_features, dim=0) # (batch_size, n_cls, dim) or (1, n_cls, dim) depending on whether prompt_learner mode is 'dual' or 'single'

            cur_train_text_features = train_text_features
            cur_train_text_features_norm = nn.functional.normalize(cur_train_text_features, dim=-1)

            # Calculate open-set classification output logits
            if cur_train_text_features_norm.dim() == 3:
                classify_outputs = torch.einsum('ni,nci->nc', outputs_norm, cur_train_text_features_norm)
            elif cur_train_text_features_norm.dim() == 2:
                classify_outputs = torch.einsum('ni,ci->nc', outputs_norm, cur_train_text_features_norm)
            else:
                raise NotImplementedError

            if clip_align_image_classification:
                classify_outputs = classify_outputs / args.temperature # temperature
                # L-cls
                loss = criterion(classify_outputs, targets)

                if clip_align_proximal_text_num > 0:
                    # L-vlalign
                    if cur_train_text_features_norm.dim() == 3:
                        classify_outputs_clip = torch.einsum('ni,nci->nc', outputs_clip_norm, cur_train_text_features_norm)
                    elif cur_train_text_features_norm.dim() == 2:
                        classify_outputs_clip = torch.einsum('ni,ci->nc', outputs_clip_norm, cur_train_text_features_norm)

                    classify_outputs_for_align = classify_outputs
                    classify_outputs_clip_for_align = classify_outputs_clip
                    targets_for_align = targets
                    if clip_filter_out_wrong_alignment:
                        # Filter out images whose teacher visual features are misaligned with their language labels
                        clip_correct_bool = (classify_outputs_clip_for_align.argmax(dim=-1) == targets_for_align)
                        classify_outputs_for_align = classify_outputs_for_align[clip_correct_bool] # [N', C]
                        classify_outputs_clip_for_align = classify_outputs_clip_for_align[clip_correct_bool]
                        targets_for_align = targets_for_align[clip_correct_bool]

                    classify_outputs_clip_for_align = classify_outputs_clip_for_align / args.temperature # temperature

                    if clip_align_text_only_proximal: # remove L-cls and only keep L-vlalign
                        loss = 0.0

                    # Get top-k language ids for each image
                    classify_outputs_clip_for_align_topk_values, classify_outputs_clip_for_align_topk_ids = classify_outputs_clip_for_align.topk(
                        k=min(clip_align_proximal_text_num, classify_outputs_clip_for_align.shape[-1]), dim=-1) # [N, K]
                    classify_outputs_for_align_topk_values = classify_outputs_for_align.gather(-1, classify_outputs_clip_for_align_topk_ids) # [N, K]
                    loss = loss + 1.0 * (F.softmax(classify_outputs_clip_for_align_topk_values, dim=-1)
                                * (F.log_softmax(classify_outputs_clip_for_align_topk_values, dim=-1) - F.log_softmax(classify_outputs_for_align_topk_values, dim=-1)
                                )).sum(dim=-1).mean()

                if clip_align_image_aux_caption:
                    # L-cap
                    aux_classify_outputs = torch.einsum('ni,ci->nc', outputs_norm, aux_clip_caption_norm)
                    contrastive_invalid_matrix = (targets[:, None] == targets[None, :])
                    tmp = torch.arange(contrastive_invalid_matrix.size(0), device=contrastive_invalid_matrix.device)
                    contrastive_invalid_matrix[tmp, tmp] = False
                    aux_classify_outputs[contrastive_invalid_matrix] = -1e7
                    arange = torch.arange(aux_classify_outputs.shape[0], device=aux_classify_outputs.device)
                    loss += criterion(aux_classify_outputs, arange)

            if (clip_align_image_mse or clip_align_image_mse_unnorm
                or clip_align_image_contrastive):
                # Teacher-student visual space alignment

                # For visual space alignment, filtering out images misaligned with language labels hurts performance, so we preserve the entirety of teacher' visual space
                # if clip_filter_out_wrong_alignment:
                #     classify_outputs_clip = torch.einsum('ni,ci->nc', outputs_clip_norm, cur_train_text_features_norm)
                #     clip_correct_bool = (classify_outputs_clip.argmax(dim=-1) == targets)
                # else:
                #     clip_correct_bool = torch.ones_like(targets, dtype=torch.bool)
                clip_correct_bool = torch.ones_like(targets, dtype=torch.bool)

                cur_temp = args.temperature
                if clip_align_image_mse:
                    # L-mse
                    aux_mse_loss = ((outputs_norm - outputs_clip_norm) ** 2).sum(dim=-1)[clip_correct_bool].mean()
                if clip_align_image_mse_unnorm:
                    # L-mse (alternative)
                    aux_mse_loss = torch.log(1 + ((outputs - outputs_clip) ** 2).sum(dim=-1)[clip_correct_bool].mean() / 10.0)
                if clip_align_image_contrastive:
                    # L-im-cst
                    contrastive_mat = torch.einsum('ni,mi->nm', outputs_norm[clip_correct_bool], outputs_clip_norm[clip_correct_bool]) / cur_temp
                    contrastive_labels = torch.arange(contrastive_mat.size(0), device=contrastive_mat.device)

                    if clip_align_image_contrastive_mode == 'single':
                        aux_contrastive_loss = F.cross_entropy(contrastive_mat, contrastive_labels)
                    elif clip_align_image_contrastive_mode == 'bidirectional':
                        aux_contrastive_loss = 0.5 * (
                            F.cross_entropy(contrastive_mat, contrastive_labels)
                            + F.cross_entropy(contrastive_mat.T, contrastive_labels)
                        )
                    else:
                        raise NotImplementedError()
        else:
            classify_outputs = outputs
            loss = criterion(classify_outputs, targets)

        # measure accuracy and record loss
        if loss is not None:
            losses.update(loss.item(), inputs.size(0))
        if aux_mse_loss is not None:
            aux_mse_losses.update(aux_mse_loss.item(), inputs.size(0))
        if aux_contrastive_loss is not None:
            aux_contrastive_losses.update(aux_contrastive_loss.item(), inputs.size(0))

        # Calculate average accuracy per class
        if avg_accuracy_per_class is None:
            avg_accuracy_per_class = [[0.0, 0.0] for _ in range(classify_outputs.shape[1])]
        for i in range(classify_outputs.shape[1]):
            outputs_this_class = classify_outputs[targets == i]
            if outputs_this_class.shape[0] > 0:
                avg_accuracy_per_class[i][0] += (outputs_this_class.argmax(dim=1) == i).sum().item()
                avg_accuracy_per_class[i][1] += outputs_this_class.shape[0]
        prec1 = accuracy(classify_outputs.data, targets.data, topk=(1,))[0]
        top1.update(prec1.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        #this was the original loss, corrected since it was addressed as float and needed to be addressed as tensor
        # loss_for_bkward = 0.0
        # if loss is not None:
        #     loss_for_bkward += loss
        # if aux_mse_loss is not None:
        #     loss_for_bkward += aux_mse_loss
        # if aux_contrastive_loss is not None:
        #     loss_for_bkward += aux_contrastive_loss
        # Compute loss_for_bkward as a tensor
        loss_for_bkward = 0.0
        if loss is not None:
            loss_for_bkward += loss
        if aux_mse_loss is not None:
            loss_for_bkward += aux_mse_loss
        if aux_contrastive_loss is not None:
            loss_for_bkward += aux_contrastive_loss

        # Update total_losses
        if isinstance(loss_for_bkward, torch.Tensor):
            total_losses.update(loss_for_bkward.item(), inputs.size(0))
        else:
            total_losses.update(loss_for_bkward, inputs.size(0))

        # Backward and optimize
        if isinstance(loss_for_bkward, torch.Tensor):
            loss_for_bkward.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Tot Loss: {total_loss:.4f} | Classification loss: {loss:.4f} | L-mse: {aux_mse_loss:.4f} | L-im-cst: {aux_contrastive_loss:.4f} | Temperature: {temp:.4f} | top1 (non-cls-avg): {top1: .4f} '.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    total_loss=total_losses.avg,
                    loss=losses.avg if loss is not None else 0.0,
                    aux_mse_loss=aux_mse_losses.avg if aux_mse_loss is not None else 0.0,
                    aux_contrastive_loss=aux_contrastive_losses.avg if aux_contrastive_loss is not None else 0.0,
                    temp=args.temperature,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()

    avg_accuracy_per_class = [100.0 * x[0] / x[1] for x in avg_accuracy_per_class if x[1] > 0]
    print("Average accuracy per class: {}".format(avg_accuracy_per_class))
    mean_acc = np.mean(avg_accuracy_per_class)
    print("Mean accuracy per class {}".format(mean_acc))

    return (total_losses.avg, mean_acc)

def test(val_loader, model, criterion, epoch, use_cuda, 
         val_transform, val_text_features=None, clip_model=None, clip_preprocess=None, clip_device=None,
         target_remap=None, 
         few_shot_features=None, support_set_idx=None, prompt_learner=None, text_encoder=None, prompt_mode='test'):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    avg_accuracy_per_class = None
            
    tot_idx = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # If performing retrieval-based few-shot learning, remove few-shot examples from the test set
        if few_shot_features is not None:
            assert support_set_idx is not None
            # remove few-shot examples from the test set
            keep_bool = ~torch.isin((tot_idx + torch.arange(targets.shape[0])), support_set_idx)
            inputs = inputs[keep_bool]
            targets = targets[keep_bool] 
        tot_idx += targets.shape[0]
        
        inputs = val_transform(inputs)

        if use_cuda:
            inputs, targets = inputs.to(cuda_device), targets.to(cuda_device)
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        with torch.no_grad():
            outputs = model(inputs)
            cur_val_text_features = val_text_features
            
        if clip_model is not None:
            outputs_norm = nn.functional.normalize(outputs, dim=1)
            ## input: normalized clip image features, output: normalized clip text features
            if prompt_learner is not None and text_encoder is not None:
                with torch.no_grad():
                    if prompt_mode == 'train':
                        tokenized_prompts = prompt_learner.train_tokenized_prompts
                    elif prompt_mode == 'test':
                        tokenized_prompts = prompt_learner.val_tokenized_prompts
                    elif prompt_mode == 'val_on_train':
                        tokenized_prompts = prompt_learner.val_on_train_tokenized_prompts
                    else: 
                        raise NotImplementedError()
                    prompts = prompt_learner(outputs_norm.to(cuda_device), mode=prompt_mode)
                    cur_val_text_features = []
                    for pts_i in prompts: # (n_cls, n_tkn, ctx_dim)
                        minib = 128
                        tokenized_idx = 0
                        cur_val_text_feature = []
                        while tokenized_idx < len(tokenized_prompts):
                            cur_val_text_feature.append(
                                text_encoder(pts_i[tokenized_idx:tokenized_idx+minib], tokenized_prompts[tokenized_idx:tokenized_idx+minib])
                            )
                            tokenized_idx += minib
                        cur_val_text_feature = torch.cat(cur_val_text_feature, dim=0)
                        cur_val_text_features.append(cur_val_text_feature[None, ...])
                    cur_val_text_features = torch.cat(cur_val_text_features, dim=0) # [B, n_cls, d]
                    cur_val_text_features_norm = nn.functional.normalize(cur_val_text_features, dim=-1)
                    classify_outputs = torch.einsum('ni,nci->nc', outputs_norm, cur_val_text_features_norm)
            else:
                cur_val_text_features_norm = nn.functional.normalize(cur_val_text_features, dim=-1)
                classify_outputs = torch.einsum('ni,ci->nc', outputs_norm, cur_val_text_features_norm)
            if few_shot_features is None:
                classify_outputs = classify_outputs / args.temperature # temperature
            else:
                # few-shot following TIP-adapter
                beta = 5.5
                alpha = 1.0
                A = torch.exp(-beta * (1 - torch.einsum('ni,cmi->ncm', outputs_norm, few_shot_features))) # [B, n_class, few_shot_num]
                classify_outputs = classify_outputs + alpha * A.sum(dim=-1)
        else:
            classify_outputs = outputs
            if target_remap is not None:
                classify_outputs = classify_outputs[:, target_remap[0]:target_remap[1]]
            
        # Calculate validation loss
        if classify_outputs.ndim == 2:
            loss = criterion(classify_outputs, targets)
        elif classify_outputs.ndim == 3:
            # classify_outputs shape [B, C, n_experts]
            loss = criterion(classify_outputs, targets[:, None].tile(1, classify_outputs.shape[-1]))
            classify_outputs_amax = classify_outputs.argmax(dim=1)
            tmp = torch.zeros_like(classify_outputs[:, :, 0])
            ones = torch.ones_like(classify_outputs_amax).float()
            tmp.scatter_(dim=1, index=classify_outputs_amax, src=ones, reduce='add')
            classify_outputs = tmp # [B, C]
            
        # Measure accuracy and record loss
        prec1 = accuracy(classify_outputs.data, targets.data, topk=(1,))[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
            
        # Calculate average accuracy per class
        if avg_accuracy_per_class is None:
            avg_accuracy_per_class = [[0.0, 0.0] for _ in range(classify_outputs.shape[1])]
        for i in range(classify_outputs.shape[1]):
            classify_outputs_this_class = classify_outputs[targets == i]
            if classify_outputs_this_class.shape[0] > 0:
                avg_accuracy_per_class[i][0] += (classify_outputs_this_class.argmax(dim=1) == i).sum().item()
                avg_accuracy_per_class[i][1] += classify_outputs_this_class.shape[0]

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1 (non-cls-avg): {top1: .4f} '.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    
    avg_accuracy_per_class = [100.0 * x[0] / x[1] for x in avg_accuracy_per_class if x[1] > 0]
    print("Average accuracy per class: {}".format(avg_accuracy_per_class))
    mean_acc = np.mean(avg_accuracy_per_class)
    print("Mean accuracy per class {}".format(mean_acc))
    return (losses.avg, mean_acc)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth'))

def adjust_learning_rate(optimizer, epoch, scheduler=None):
    global state
    if epoch in args.schedule and scheduler is None:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
    if scheduler is not None and epoch > 0:
        scheduler.step()

if __name__ == '__main__':
    main()
