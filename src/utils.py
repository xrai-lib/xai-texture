import subprocess
import platform
import torch
import pandas as pd
import os

#clear the terminal screen
def clear_screen():
    # Check if the operating system is Windows
    if platform.system() == "Windows":
        subprocess.run("cls", shell=True, check=True)
    else:
        # Assume the operating system is Unix/Linux/Mac
        subprocess.run("clear", shell=True, check=True)

# Helper function to calculate IoU
def calculate_iou(preds, labels):
    preds = preds > 0.5  # Threshold the predictions to get binary output
    labels = labels > 0.5
    intersection = (preds & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (preds | labels).float().sum((1, 2))         # Will be zero if both are 0

    iou = (intersection + 1e-6) / (union + 1e-6)  # We smooth our division to avoid 0/0
    return torch.mean(iou).item()  # Return the mean IoU score for the batch

def add_to_results(file_path, dataset, value):
     # Check if the CSV file exists
    if os.path.isfile(file_path):
        # Read the existing CSV file
        df = pd.read_csv(file_path)
    else:
        # Create a new DataFrame if the file does not exist
        df = pd.DataFrame(columns=['Dataset', 'IOU Accuracy'])
    
    # Check if the row to update exists (assuming 'id' is the column name for the row identifier)
    if dataset in df['Dataset'].values:
        # Update the existing row
        df.update([dataset, value])
    else:
        # Append a new row
        df = df._append({'Dataset': dataset, 'IOU Accuracy': value}, ignore_index=True)
    
    # Write the DataFrame to CSV
    df.to_csv(file_path, index=False)