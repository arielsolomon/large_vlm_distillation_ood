import torch
import torchvision
import numpy as np

from pycocotools import mask as coco_mask

def get_coco_api_from_dataset(dataset):
    for i in range(len(dataset)):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    raise TypeError("Dataset is not of type CocoDetection")

def convert_to_coco_api(ds):
    coco_ds = get_coco_api_from_dataset(ds)
    return coco_ds

def coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(annot):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in annot)

    def _count_visible_keypoints(annot):
        return sum(sum(1 for v in obj["keypoints"][2::3] if v > 0) for obj in annot)

    min_keypoints_per_image = 10

    if not isinstance(dataset, torchvision.datasets.CocoDetection):
        raise TypeError("dataset should be of type CocoDetection, but got {}".format(type(dataset)))

    ids = []
    for ds_idx in range(len(dataset)):
        ann_ids = dataset.coco.getAnnIds(imgIds=ds_idx, iscrowd=None)
        anns = dataset.coco.loadAnns(ann_ids)
        if len(anns) == 0:
            continue

        if _has_only_empty_bbox(anns):
            continue

        if "keypoints" not in anns[0]:
            ids.append(ds_idx)
            continue

        if _count_visible_keypoints(anns) >= min_keypoints_per_image:
            ids.append(ds_idx)

    dataset.ids = ids
    return dataset

def convert_coco_poly_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks
