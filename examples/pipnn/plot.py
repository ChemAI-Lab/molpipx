import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the directory containing the files
directory_path = '/scratch/r/ravh011/ravh011/pipjax_test/pipnn_results_l0E'

# List all files in the directory that match the pattern
files = [f for f in os.listdir(directory_path) if f.startswith(
    "training_trajectory_ema_wgrad_l0_")]

# Initialize lists to store hyper-parameter values and corresponding loss values
hyper_params = []
loss_collections = []

# Process each file
for file in files:
    # Extract the hyper-parameter value from the filename
    hyper_param = float(file.split('_l0_')[-1].replace('.csv', ''))

    # Read the data from the file
    df = pd.read_csv(os.path.join(directory_path, file))

    # Check for the necessary column
    if 'val_e_loss' not in df.columns:
        raise ValueError(f"Column 'val_f_loss' not found in {file}")

    # Extract the 25 lowest values from the 'val_e_loss' column
    lowest_losses = df['val_e_loss'].nsmallest(100).reset_index(drop=True)

    # Store the hyper-parameter and corresponding losses
    hyper_params.append(hyper_param)
    loss_collections.append(lowest_losses)

# Create a DataFrame for plotting
plot_data = pd.DataFrame({
    'Hyper-parameter': [hp for hp, losses in zip(hyper_params, loss_collections) for _ in range(len(losses))],
    'Loss': [loss for losses in loss_collections for loss in losses]
})

# Plotting
plt.figure()  # figsize = (10, 6)

sns.boxplot(x='Hyper-parameter', y='Loss', data=plot_data)
plt.xlabel(r"$\lambda_{0}$", fontsize=20)

plt.text(0.0002, 0.00053, 'a)', fontsize=20)
plt.ylabel(r"${\cal L}^{E}_{\text{val}}(\lambda_{0})$", fontsize=22)
plt.ylim(0.00025, 0.00056)

# plt.text(0.001, 1.2, 'b)', fontsize=20)
# plt.ylabel(r"${\cal L}^{F}_{\text{val}}(\lambda_{0})$", fontsize=22)
# # plt.ylim(0, 12)

plt.xticks(rotation=45)
plt.tight_layout()
save_dir = '/scratch/r/ravh011/ravh011/pipjax_test/'
plt.savefig(f'{save_dir}test_e.png', dpi=600)
