import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

root = '/Data/federated_learning/large_vlm_distillation_ood/Resnet18_classification/'
acc_file = root+'classify_with_add_OOD_and_ind_accuracy_12_07_gpu2.txt'


with open(acc_file, 'r') as f:
    lines = f.readlines()
i = 3
train_accuracies, test_accuracies_ood, test_accuracies_ind = [], [], []
for i, line in enumerate(lines[2:]):
    if (i % 3 == 0):
        train_accuracies.append(line.split('acc: ', 1)[1][:-2])
    elif (i % 3 == 1):
        test_accuracies_ood.append(line.split('acc: ', 1)[1][:-2])
    else:
        test_accuracies_ind.append(line.split('acc_ind: ', 1)[1][:-2])
df = pd.DataFrame(columns=['Train_acc','Test_acc_ood','Test_acc_ind'])
df['Train_acc'] = train_accuracies
df['Test_acc_ood'] = test_accuracies_ood
df['Test_acc_ind'] = test_accuracies_ind

df['added_samples'] = np.arange(0,len(train_accuracies)*5,5)
df['added_samples'] = pd.to_numeric(df['added_samples'], errors='coerce')
df['Train_acc'] = pd.to_numeric(df['Train_acc'], errors='coerce')
df['Test_acc_ood'] = pd.to_numeric(df['Test_acc_ood'], errors='coerce')
df['Test_acc_ind'] = pd.to_numeric(df['Test_acc_ind'], errors='coerce')

x = df['added_samples']
plt.scatter(x,df['Test_acc_ood'], label='Test Test_acc_ood')
plt.scatter(x,df['Test_acc_ind'], label='Test Test_acc_ind')
plt.xlabel('Num samples added')
plt.ylabel('Accuracy[%]')
plt.ylim(0,100)
plt.legend()
plt.tight_layout()
plt.title('Training with adding few shots (5 at a time) OOD at gpu2')
plt.savefig(root+'accuracies_OOD_Ind_from_gpu2_12_07.png')
plt.show()