import pandas as pd
import os

directory = "/Users/oldrichpriklenkiv/untitled folder 13.45.33/downloaded_files"

import os
import pandas as pd
import chardet  # To detect encoding issues

import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn


def detect_encoding(file_path, num_lines=1000):
    """Detects file encoding using chardet."""
    with open(file_path, "rb") as f:
        raw_data = f.read(num_lines)
    return chardet.detect(raw_data)["encoding"]

def load_csv_files_to_dataframe(directory):
    """
    Reads and concatenates CSV files sequentially, extracting only:
    - Column 0 (ISIN / Stock ID)
    - Column 3 (Date, formatted correctly)
    - Column 4 (Closing Price, converted to float)
    """

    all_dataframes = []

    # Get a list of CSV files (sorted alphabetically or by timestamp)
    csv_files = sorted([f for f in os.listdir(directory) if f.endswith('.csv')])

    if not csv_files:
        print("No CSV files found.")
        return None

    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)
        
        # **Detect encoding first**
        detected_encoding = detect_encoding(file_path)
        print(f" Detected encoding for {csv_file}: {detected_encoding}")

        try:
            # **Read only necessary columns (0, 3, 4)**
            df = pd.read_csv(file_path, encoding=detected_encoding, usecols=[0, 3, 4])
        except UnicodeDecodeError:
            print(f" Encoding error in {csv_file}, trying fallback encoding (ISO-8859-1)...")
            try:
                df = pd.read_csv(file_path, encoding='ISO-8859-1', usecols=[0, 3, 4])
            except Exception as e:
                print(f" Failed to read {csv_file} due to encoding issues: {e}")
                continue  # Skip this file if all encoding methods fail

        # Rename columns for clarity
        df.columns = ["ISIN", "Date", "Closing Price"]

        # ✅ Convert Date column to proper format
        df["Date"] = pd.to_datetime(df["Date"], format="%Y/%m/%d", errors="coerce")

        # ✅ Convert Closing Price to numeric (float)
        df["Closing Price"] = pd.to_numeric(df["Closing Price"], errors="coerce")

        # Track the source file (optional)
        df["source_file"] = csv_file

        all_dataframes.append(df)

    # **Sequentially merge** all DataFrames row-wise
    if all_dataframes:
        final_df = pd.concat(all_dataframes, ignore_index=True)
        print("\n Successfully extracted and concatenated necessary columns.\n")

        #  **Print debug info**
        print(" First 5 rows:\n", final_df.head())  # Show first few rows
        print("\n Last 5 rows:\n", final_df.tail())  # Show last few rows
        print("\n Unique source files included:", final_df["source_file"].unique())  # Show unique file sources
        print("\n Total rows:", len(final_df))  # Show total rows in final DataFrame

        return final_df
    else:
        print("No valid DataFrames were created.")
        return None

# Load and concatenate CSV files
final_df = load_csv_files_to_dataframe(directory)


def transform_dataframe_to_tensor(final_df):
    """
    Converts stock data DataFrame into a tensor of shape (N, T, F).
    
    - N: Number of unique stocks (companies) from the 'ISIN' column.
    - T: Maximum time steps (dates) per company (padded if necessary).
    - F: Features (1 feature: 'Closing Price').

    Returns:
    - company_tensor (Tensor): Shape [N, T, 1]
    - company_names (List[str]): List of unique stock ISINs.
    - edge_index (Tensor): Defines company relationships in the graph.
    """
    
    # Ensure the DataFrame is sorted by Date
    final_df = final_df.sort_values(by=["ISIN", "Date"])

    # Get unique company identifiers (ISINs)
    company_names = sorted(final_df["ISIN"].unique())

    # Dictionary to store each company's time-series tensor
    company_data = {}

    for company in company_names:
        # Extract time-series data for this company
        company_df = final_df[final_df["ISIN"] == company]

        # Get Closing Price as a PyTorch tensor
        closing_prices = torch.tensor(company_df["Closing Price"].values, dtype=torch.float32)

        # Store it in a dictionary
        company_data[company] = closing_prices

    # Determine the maximum sequence length (T)
    max_T = max(len(prices) for prices in company_data.values())

    # Create a list of padded tensors (ensuring equal time steps for all companies)
    padded_tensors = []
    for company in company_names:
        tensor = company_data[company]
        padded_tensor = torch.nn.functional.pad(tensor, (0, max_T - tensor.shape[0]))  # Right-pad with 0s
        padded_tensors.append(padded_tensor)

    # Stack tensors into a final shape of [N, T, 1] (1 feature: Closing Price)
    company_tensor = torch.stack(padded_tensors).unsqueeze(-1)  # Add feature dimension

    # Create a simple fully connected graph (edge_index)
    N = len(company_names)
    edge_list = [[i, j] for i in range(N) for j in range(N) if i != j]  # No self-loops
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # Shape [2, num_edges]

    return company_tensor, company_names, edge_index

# Convert DataFrame to tensor
company_tensor, company_names, edge_index = transform_dataframe_to_tensor(final_df)

# Print debug information
print(f"✅ Tensor shape: {company_tensor.shape}  # Expected [N, T, 1]")
print(f"✅ Number of unique companies (N): {len(company_names)}")
print(f"✅ Edge Index shape: {edge_index.shape}  # Expected [2, num_edges]")
print(f"✅ Example company data: \n{company_tensor[:1]}")  # Show first 2 companies' data


class TemporalGNN(nn.Module):
    def __init__(self, in_channels):
        """
        Initializes a Temporal GNN for stock price prediction.
        """
        super(TemporalGNN, self).__init__()
        # Initialize a GCNConv layer
        self.conv = GCNConv(in_channels, in_channels)
        print("✅ Initialized TemporalGNN with a GCNConv layer.")

    def forward(self, company_tensor, edge_index):
        """
        Forward pass for the Temporal GNN. Debug statements are included
        to print out the shape and content of data at each step.
        """
        # Get dimensions: N = number of companies, T = time steps, F = features
        N, T, F = company_tensor.size()
        print(f"\n[Forward Start] Input company_tensor shape: {company_tensor.shape}")
        print(f"[Forward Start] Edge index shape: {edge_index.shape}")
        print(f"[Forward Start] Edge index:\n{edge_index}")

        updated_features = []

        # Process each time step separately
        for t in range(T):
            # Extract features for time step t (shape: [N, F])
            x_t = company_tensor[:, t, :]
            print(f"\n🔹 [Time Step {t}] Original x_t shape: {x_t.shape}")
            print(f"🔹 [Time Step {t}] x_t values:\n{x_t}")

            # Reshape if necessary (ensuring shape [N, F])
            x_t = x_t.view(N, -1)
            print(f"🔹 [Time Step {t}] Reshaped x_t for GCNConv: {x_t.shape}")

            # Apply the GCN convolution for time step t
            x_t_updated = self.conv(x_t, edge_index)
            print(f"🔹 [Time Step {t}] Output from GCNConv (before activation):")
            print(f"    Type: {type(x_t_updated)}")
            print(f"    Values:\n{x_t_updated}")

            # Verify that the output is a tensor
            if not isinstance(x_t_updated, torch.Tensor):
                raise ValueError(f"❌ GCNConv output is not a tensor at t={t}, got {type(x_t_updated)}")

            # Apply activation (ReLU)
            x_t_updated = F.relu(x_t_updated)
            print(f"🔹 [Time Step {t}] After ReLU activation: shape {x_t_updated.shape}")
            print(f"    Values:\n{x_t_updated}")

            # Add time dimension back and store updated features
            updated_features.append(x_t_updated.unsqueeze(1))

        # Concatenate features from all time steps -> shape [N, T, F]
        out = torch.cat(updated_features, dim=1)
        print(f"\n[Forward End] Concatenated output shape: {out.shape}")
        return out

# Example Usage
if __name__ == "__main__":
    # Convert DataFrame to tensor
    company_tensor, company_names, edge_index = transform_dataframe_to_tensor(final_df)

    # Initialize the model with F=1 (Closing Price is the only feature)
    model = TemporalGNN(in_channels=1)

    # Forward pass
    updated_company_data = model(company_tensor, edge_index)

    # Print output shapes
    print("\n🔹 Final Output Data:")
    print(f"Input shape: {company_tensor.shape}")         # Expected: [N, T, 1]
    print(f"Output shape: {updated_company_data.shape}")  # Expected: [N, T, 1]

    