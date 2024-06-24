import pandas as pd

src = '/Data/federated_learning/large_vlm_distillation_ood/faster_Rcnn/data/coco_4_vlm/data/train.txt'
df = pd.read_csv(src, header=None)
def df_to_dict_with_apostrophes(df):
    result = {}
    for entry in df[0]:
        key, value = entry.split(': ')
        result[int(key)] = f"'{value}'"
    return result

result_dict = df_to_dict_with_apostrophes(df)
with open(src[:-9]+'result_dict.txt', 'w') as file:
    for key, value in result_dict.items():
        file.write(f"{key}: {value}\n")