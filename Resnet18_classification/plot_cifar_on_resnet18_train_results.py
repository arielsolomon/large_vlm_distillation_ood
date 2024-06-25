import pandas as pd
import matplotlib.pyplot as plt

root = '/home/user1/ariel/fed_learn/large_vlm_distillation_ood/resnet18_cifar_classification/'

def plot_scatter(src, is_test=False):
# Create the scatter plot


        if is_test:
            df = pd.read_csv(src+'corruption_accuracies_23_06_add_test.txt', header=None)
            print(df.head(3))
            df['corruption_name'] = df[1]
            df['corruption_accuracy'] = df[3]
            df['train accuracy'] = df[5]
            df['test_accuracy'] = df[7]
            df['train_accuracy'] = pd.to_numeric(df['train accuracy'])
            df['corruption_accuracy'] = pd.to_numeric(df['corruption_accuracy'])
            df['test_accuracy'] = pd.to_numeric(df['test_accuracy'])

            # Extract data for plotting
            corruption_names = df['corruption_name'].tolist()
            train_accuracy = df['train_accuracy'].tolist()
            corruption_accuracy = df['corruption_accuracy'].tolist()
            test_accuracy = df['test_accuracy'].tolist()
            plt.figure(figsize=(10, 6))  # Adjust figure size as needed

            # Scatter plot with different colors for each variable
            plt.scatter(corruption_names, train_accuracy, color='blue', label='Train accuracy')
            plt.scatter(corruption_names, corruption_accuracy, color='red', label='Corrupted accuracy')
            plt.scatter(corruption_names, test_accuracy, color='green', label='Test accuracy')

            plt.xticks(range(len(corruption_names)), corruption_names, rotation=45, ha='right')
            # Add labels and title
            plt.xlabel('Corruption Type')
            plt.ylabel('Accuracy')
            plt.title('Train vs. Corruption Accuracy')

            # Customize plot (optional)
            # You can add grid lines, adjust marker sizes, etc.
            plt.grid(True)

            # Save the plot
            plt.savefig('corruption_accuracy_plot_with_test_23_06.png')  # Change filename as desired
            plt.legend()
            # Show the plot (optional)
            plt.show()
        else:


            df = pd.read_csv(src+'corruption_accuracies_no_test_23_06.txt', header=None)
            print(df.head(3))
            df['corruption_name'] = df[1]

            df[['meaningless1','meaningless2','meaningless3','corruption_accuracy']]=df[2].str.split(' ')
            df['corruption_accuracy'] = df[2].str.split(' ')[0][3]
            df['train accuracy'] = df[5]
            df['test_accuracy'] = df[7]
            df['train_accuracy'] = pd.to_numeric(df['train accuracy'])
            df['corruption_accuracy'] = pd.to_numeric(df['corruption_accuracy'])
            df['test_accuracy'] = pd.to_numeric(df['test_accuracy'])

            # Extract data for plotting
            corruption_names = df['corruption_name'].tolist()
            train_accuracy = df['train_accuracy'].tolist()
            corruption_accuracy = df['corruption_accuracy'].tolist()
            test_accuracy = df['test_accuracy'].tolist()
            plt.figure(figsize=(10, 6))  # Adjust figure size as needed

            # Scatter plot with different colors for each variable
            plt.scatter(corruption_names, train_accuracy, color='blue', label='Train accuracy')
            plt.scatter(corruption_names, corruption_accuracy, color='red', label='Corrupted accuracy')
            if is_test == True:
                plt.scatter(corruption_names, test_accuracy, color='green', label='Test accuracy')

                plt.xticks(range(len(corruption_names)), corruption_names, rotation=45, ha='right')
                # Add labels and title
                plt.xlabel('Corruption Type')
                plt.ylabel('Accuracy')
                plt.title('Train vs. Corruption Accuracy')

                # Customize plot (optional)
                # You can add grid lines, adjust marker sizes, etc.
                plt.grid(True)

                # Save the plot
                plt.savefig('corruption_accuracy_plot_20_06.png')  # Change filename as desired
                plt.legend()
                # Show the plot (optional)
                plt.show()

plot_scatter(root, is_test=False)