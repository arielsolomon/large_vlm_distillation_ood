import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

root = '/Data/federated_learning/large_vlm_distillation_ood/Resnet18_classification/'
acc_file = root+'Resnet_few_shots_OOD.csv'

df = pd.read_csv(acc_file,names=['Train_acc','Test_acc'])


len_col = df.shape[0]
num_samples = np.arange(0, 5*len_col, 5)
df['added_samples'] = num_samples
df['added_samples'] = pd.to_numeric(df['added_samples'], errors='coerce')
df['Train_acc'] = pd.to_numeric(df['Train_acc'], errors='coerce')
df['Test_acc'] = pd.to_numeric(df['Test_acc'], errors='coerce')

x = df['added_samples']
plt.scatter(x,df['Test_acc'], label='Test accuracy')
plt.xlabel('Num samples added')
plt.ylabel('Accuracy[%]')
plt.legend()
plt.tight_layout()
plt.title('Training with adding few shots (5 at a time) OOD')
plt.savefig(root+'accuracies_with_adding_OOD.png')
plt.show()