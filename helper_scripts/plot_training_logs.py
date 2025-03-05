import matplotlib.pyplot as plt
import json
import numpy as np
import os
import seaborn as sns
import json
import os
import matplotlib.pyplot as plt




def plot_loss_individual_logs(keys_to_plot, keys_to_title, log_files, colors):
    """
    For each log file, plot all requested keys on separate subplots.

    Args:
        keys_to_plot (list): A list of keys (strings) to extract from the JSON logs.
        keys_to_title (dict): A dictionary mapping each key to a title for the plot.
        log_files (list): A list of JSON log filenames to read.
        colors (list): A list of colors to use for plotting each log file.
    """
    for i, log_file in enumerate(log_files):
        # Read the entire log file once
        with open(log_file, 'r') as f:
            data = [json.loads(line.strip()) for line in f]

        for key in keys_to_plot:
            # Extract values for this key
            values = [entry[key] for entry in data]

            # Create a new figure for each key
            plt.figure(figsize=(8, 6))

            # Plot
            plt.plot(values, linewidth=1)
            plt.title(keys_to_title[key], fontsize=16, fontweight='bold')
            plt.xlabel("Epoch", fontsize=16, fontweight='bold')
            plt.ylabel("Loss", fontsize=16, fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.6)

            # save the plot using base path from log file and key name using keys_to_title
            base_save_path = os.path.dirname(log_file)
            filename = keys_to_title[key].replace(" ", "_").lower()
            plt.savefig(os.path.join(base_save_path, f"{filename}.png"), dpi=300, bbox_inches='tight')
            # Display the figure
            #plt.show()
            plt.close()


def plot_loss_combined_logs(keys_to_plot, keys_to_title, log_files, log_files_to_title, colors):
    """
    For each key, combine data from all logs into a single plot.

    Args:
        keys_to_plot (list): A list of keys (strings) to extract from the JSON logs.
        keys_to_title (dict): A dictionary mapping each key to a title for the plot.
        log_files (list): A list of JSON log filenames to read.
        colors (list): A list of colors to use for plotting each log file.
    """
    for key in keys_to_plot:
        plt.figure(figsize=(8, 6))

        for i, log_file in enumerate(log_files):
            with open(log_file, 'r') as f:
                data = [json.loads(line.strip()) for line in f]

            # Extract values for this key
            values = [entry[key] for entry in data]

            # Label will be derived from log_file name
            plt.plot(values, color=colors[i], label=log_files_to_title[i], linewidth=1)

        plt.title(keys_to_title[key], fontsize=16, fontweight='bold')
        plt.xlabel("Epoch", fontsize=16, fontweight='bold')
        plt.ylabel("Loss", fontsize=16, fontweight='bold')

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        for log_file in log_files:

            # save the plot using base path from log file and key name using keys_to_title
            base_save_path = os.path.dirname(log_file)
            filename = keys_to_title[key].replace(" ", "_").lower()
            filename =  "".join([f"{title}_" for title in log_files_to_title]) + filename
            plt.savefig(os.path.join(base_save_path, f"{filename}.png"), dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()



if __name__ == '__main__':

    keys_to_plot = ['train_clean_loss', 'train_clean_patient_loss', 'train_clean_slide_loss', 'train_clean_patch_loss',
                    'train_adv_loss', 'train_adv_patient_loss', 'train_adv_slide_loss', 'train_adv_patch_loss']

    keys_to_title = {"train_clean_loss": "Clean Loss", "train_clean_patient_loss": "Clean Patient Loss",
                     "train_clean_slide_loss": "Clean Slide Loss", "train_clean_patch_loss": "Clean Patch Loss",
                     "train_adv_loss": "Adversarial Loss", "train_adv_patient_loss": "Adversarial Patient Loss",
                     "train_adv_slide_loss": "Adversarial Slide Loss", "train_adv_patch_loss": "Adversarial Patch Loss"}

    log_files = [r"F:\Code\Projects\hidisc_pytorch\Results\Baseline\resnet50_timm_pretrained_exp18\log.txt",
                 r"F:\Code\Projects\hidisc_pytorch\Results\Baseline\resnet50_exp18\log.txt",
                 r"F:\Code\Projects\hidisc_pytorch\Results\Baseline\resnet50_at_exp18\log.txt"]

    log_files_to_titles = ["ResNet-50(ImageNet)", "ResNet-50(Scratch)", "ResNet-50(AT)"]

    colors = plt.cm.viridis(np.linspace(0, 1, len(log_files)))


    def plot_loss_from_logs(keys_to_plot, keys_to_title, log_files, colors):
        for key in keys_to_plot:
            plt.figure(figsize=(8, 6))  # Adjust figure size for better readability

            for i, log_file in enumerate(log_files):
                with open(log_file, 'r') as f:
                    values = [json.loads(line.strip())[key] for line in f.readlines()]

                label = log_file.split(".")[0]  # Extract label from filename
                plt.plot(values, color=colors[i], linewidth=2)  # Improved visibility

            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)  # Add grid for better readability
            plt.title(keys_to_title[key], fontsize=14)  # Add title to plot
            # plt.savefig(f"{key}.png", dpi=300, bbox_inches='tight')  # Save with high resolution
            plt.show()  # Display plot


    # plot_loss_from_logs(keys_to_plot, keys_to_title, log_files, colors)
    # 1. Plot individual logs
    plot_loss_individual_logs(keys_to_plot, keys_to_title, log_files, colors)

    # 2. Plot combined logs
    plot_loss_combined_logs(keys_to_plot, keys_to_title, log_files, log_files_to_titles, colors)

