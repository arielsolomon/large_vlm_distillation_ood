import argparse
from pathlib import Path
from typing import Callable, Tuple, Any
import pickle
import numpy as np
import torch
import clip
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset

from dataloaders.cifar10_loader import load_raw_data


class ProcessedCifar10(VisionDataset):
    train_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4"]

    validation_list = ["data_batch_5"]

    test_list = ["test_batch"]

    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
    ) -> None:

        super().__init__(root)

        self.train = train  # training set or test set

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            path = Path(self.root) / file_name
            with open(path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = torch.vstack(self.data)
        self._load_meta()

    def _load_meta(self) -> None:
        path = Path(self.root)
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


def get_command_line_arguments(parser):
    """
    Parse command-line arguments.

    Parameters:
    parser (argparse.ArgumentParser): Argument parser object

    Returns:
    argparse.Namespace: Parsed arguments
    """
    parser.add_argument("--data-path", type=str, default=f"{str(Path.home())}/datasets/cifar/cifar-10-batches-py",
                        help="dir path for datafolder")

    parser.add_argument("--datasets-path", type=str, default=f"{str(Path.cwd())}/saved_datasets/cifar10",
                        help="dir path for datafolder")

    parser.add_argument("--batch-size", type=int, default="1024", help="Number of images in train batch")

    args = parser.parse_args()
    return args


def main(args):
    # Load the model
    _, preprocess = clip.load('ViT-B/32')

    data_batches_path = Path(args.data_path)

    batches_paths = [b for b in data_batches_path.glob('*_batch*')]

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    datasets_path = Path(args.datasets_path)
    if not datasets_path.exists():
        datasets_path.mkdir(parents=True)

    text_all_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes])
    for batch_path in batches_paths:

        assert batch_path.exists(), f'{batch_path} does not exist'
        assert batch_path.is_file(), f'Expected a file'

        batch_dict = unpickle(batch_path)

        subset_path = datasets_path / batch_path.name
        if not subset_path.exists():
            subset_path.mkdir()

        print(f'Creating {batch_path.name}')
        original_processed_batch, ood_processed_batch = create_saved_set(subset_path, preprocess, batch_dict[b'data'])
        print(f'{batch_path.name}_set created and saved to {subset_path}')


def unpickle(file):
    with open(file, 'rb') as fo:
        batch_dict = pickle.load(fo, encoding='bytes')
    return batch_dict


def create_saved_set(path: Path, preprocess: Callable[[Image], torch.tensor], data: torch.tensor) \
        -> tuple[Tensor, Tensor]:
    original_tensor = torch.ones(size=(1, 3, 224, 224))
    ood_tensor = torch.ones(size=(1, 3, 224, 224))
    ood_trans = transforms.Compose(
        [transforms.ToTensor(),
         transforms.ColorJitter(contrast=0.5, brightness=1.0),
         transforms.ToPILImage()])
    for i in range(len(data)):
        image: np.array = data[i].reshape(3, 32, 32)
        image = image.transpose((1, 2, 0))
        pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
        image_input = preprocess(pil_image).unsqueeze(0)
        image_input_ood = preprocess(ood_trans(pil_image)).unsqueeze(0)
        original_tensor = torch.vstack([original_tensor, image_input])
        ood_tensor = torch.vstack([ood_tensor, image_input_ood])
    return original_tensor[1:], ood_tensor[1:]


# def create_saved_set(path: Path, preprocess: Callable[[Image], torch.tensor],
#                      dataset: Dataset, text_all_inputs: torch.tensor) -> None:
#     images, images_ood, texts = [], [], []
#     for image, label in dataset:
#         # print('**********   Processing image', total)
#         trans = transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.ColorJitter(contrast=0.5, brightness=1.0),
#              transforms.ToPILImage()])
#
#         image_input = preprocess(image).unsqueeze(0)
#         image_input_ood = preprocess(trans(image)).unsqueeze(0)
#         text_input = text_all_inputs[label]
#         images.append(image_input)
#         images_ood.append(image_input_ood)
#         texts.append(text_input)
#     torch.save(torch.stack(images), path / 'images.pt')
#     torch.save(torch.stack(images_ood), path / 'images_ood.pt')
#     torch.save(torch.stack(texts), path / 'texts.pt')


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Datasets")
    args = get_command_line_arguments(parser)
    main(args)
