import fiftyone.zoo as foz
import fiftyone as fo

fo.config.dataset_zoo_dir = "/home/user1/ariel/fed_learn/large_vlm_distillation_ood/coco_2017_8cls/"
classes_names = ['car','person','boat','train','dog','traffic light','airplane','bicycle']
for cls in classes_names:

    dataset = foz.load_zoo_dataset(
        "coco-2017",
        splits=["train", "test","validation"],
        label_types=["detections"],
        classes=[cls],
        max_samples=4000,
    )

# root = '/home/user1/ariel/fed_learn/datasets/coco/'