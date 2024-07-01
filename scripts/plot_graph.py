import matplotlib.pyplot as plt
import pandas as pd

path = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/larg_vlm_summary.xlsx'
df = pd.read_excel(path, header=None).iloc[1:,:]
d = {'Val_cifar_c': df.iloc[:,1], 'Val': df.iloc[:,2]}
df.iloc[:,1].plot(style='x', label='Noised')
df.iloc[:,2].plot(style='o', label='Val')

labels = df.iloc[:,0].tolist()

# Set x-axis ticks and labels using plt.xticks
plt.xticks(range(len(df)), labels,rotation=45, ha='left', fontsize=7,weight='bold')  # Use range(len(df)) for tick positions
x_label = 'noise type'
y_label = 'Accuracy[%]'
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.legend()
plt.show()
