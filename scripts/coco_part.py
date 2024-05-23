import fiftyone.zoo as foz
import fiftyone as fo
import os

#fo.config.dataset_zoo_dir = "/home/user1/ariel/fed_learn/large_vlm_distillation_ood/coco-2017/"
classes = os.listdir('/home/user1/ariel/fed_learn/large_vlm_distillation_ood/coco-2017/train/')
classes_names = [item[2:] for item in classes]
for cls in classes[5:]:

    fo.config.dataset_zoo_dir = "/home/user1/ariel/fed_learn/large_vlm_distillation_ood/coco-2017/"+cls+'/'

    if cls=='5_traffic_light':
        cls = 'trtraffic light'
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        splits=["train", "validation", "test"],
        label_types=["detections"],
        classes=[cls[2:]],
        max_samples=2000,
    )

# root = '/home/user1/ariel/fed_learn/datasets/coco/'