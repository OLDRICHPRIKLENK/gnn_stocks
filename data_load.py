import zipfile
from ftplib import FTP
import pandas as pd
from io import BytesIO
import os

from ftplib import FTP
from io import BytesIO
import os
import zipfile

def download_csv_files(ftp, current_path="/Results.ak/2024"):
    """
    Downloads all .csv files (and .csv files within .zip archives)
    from the specified directory (and any subdirectories) on the FTP server,
    returning them as a list of (file_path, BytesIO) tuples.
    """
    # Ensure we start in the desired directory
    ftp.cwd(current_path)

    csv_files = []
    items = ftp.nlst()  # List files and directories in the current directory

    for item in items:
        try:
            # Try to enter `item` as if it's a subdirectory
            ftp.cwd(item)
            print(f"Navigating into subdirectory: {item}")
            # Recursively process that subdirectory
            sub_path = f"{current_path}/{item}"
            csv_files.extend(download_csv_files(ftp, sub_path))
            # Go back to the parent directory
            ftp.cwd("..")
        except Exception:
            # If we fail to `cwd` into it, assume it is a file
            if item.endswith(".csv"):
                print(f"Downloading CSV file: {item}")
                bio = BytesIO()
                ftp.retrbinary(f"RETR {item}", bio.write)
                bio.seek(0)
                # Use the full path for reference
                file_path = f"{current_path}/{item}"
                csv_files.append((file_path, bio))

            elif item.endswith(".zip"):
                print(f"Processing ZIP file: {item}")
                bio = BytesIO()
                ftp.retrbinary(f"RETR {item}", bio.write)
                bio.seek(0)
                with zipfile.ZipFile(bio) as zf:
                    for file_name in zf.namelist():
                        if file_name.endswith(".csv"):
                            print(f"Extracting CSV from ZIP: {file_name}")
                            extracted = BytesIO(zf.read(file_name))
                            extracted.seek(0)
                            zip_file_path = f"{current_path}/{file_name}"
                            csv_files.append((zip_file_path, extracted))

    return csv_files


# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    ftp_server = "ftp.pse.cz"
    ftp = FTP(ftp_server)
    ftp.login()

    # Download all CSVs (and any CSVs within .zip) 
    # from the /Results.ak/2024 directory
    csv_files = download_csv_files(ftp, "/Results.ak/2024")
    print(csv_files)

    # `csv_files` is now a list of (file_path, BytesIO) tuples in memory.
    # You can pass these directly to your DataFrame loader:
    # df = load_csv_files_to_dataframe_from_memory(csv_files)

import pandas as pd
import chardet
from io import BytesIO

def load_csv_files_to_dataframe(csv_files):
    """
    Reads and concatenates CSV files sequentially (from in-memory BytesIO objects),
    extracting only:
      - Column 0 (ISIN / Stock ID)
      - Column 3 (Date, formatted correctly)
      - Column 4 (Closing Price, converted to float)

    Parameters
    ----------
    csv_files : list of tuples
        Each tuple is (filename, BytesIO_object) representing a CSV file in memory.

    Returns
    -------
    final_df : pd.DataFrame or None
        A concatenated DataFrame of all CSV data, or None if no valid DataFrames.
    """

    all_dataframes = []

    if not csv_files:
        print("No CSV files provided.")
        return None

    for (csv_file_name, csv_bytes) in csv_files:
        print(f"Processing file: {csv_file_name}")

        # Detect encoding from the first ~1000 bytes
        csv_bytes.seek(0)  # Ensure we start at the beginning
        raw_data = csv_bytes.read(1000)
        detected_encoding = chardet.detect(raw_data)["encoding"]
        print(f" Detected encoding for {csv_file_name}: {detected_encoding}")

        # Reset to the beginning for actual read
        csv_bytes.seek(0)

        try:
            # Read only the necessary columns (0, 3, 4)
            df = pd.read_csv(csv_bytes, encoding=detected_encoding, usecols=[0, 3, 4])
        except UnicodeDecodeError:
            print(f" Encoding error in {csv_file_name}, trying fallback encoding (ISO-8859-1)...")
            csv_bytes.seek(0)
            try:
                df = pd.read_csv(csv_bytes, encoding='ISO-8859-1', usecols=[0, 3, 4])
            except Exception as e:
                print(f" Failed to read {csv_file_name} due to encoding issues: {e}")
                continue  # Skip this file if all encoding methods fail

        # Rename columns for clarity
        df.columns = ["ISIN", "Date", "Closing Price"]

        # Convert Date column to proper format
        df["Date"] = pd.to_datetime(df["Date"], format="%Y/%m/%d", errors="coerce")

        # Convert Closing Price to numeric (float)
        df["Closing Price"] = pd.to_numeric(df["Closing Price"], errors="coerce")

        # Track the source file (optional)
        df["source_file"] = csv_file_name

        all_dataframes.append(df)

    # Merge all DataFrames row-wise
    if all_dataframes:
        final_df = pd.concat(all_dataframes, ignore_index=True)
        print("\n Successfully extracted and concatenated necessary columns.\n")

        # Debug info
        print(" First 5 rows:\n", final_df.head())
        print("\n Last 5 rows:\n", final_df.tail())
        print("\n Unique source files included:", final_df["source_file"].unique())
        print("\n Total rows:", len(final_df))

        return final_df
    else:
        print("No valid DataFrames were created.")
        return None

final_df = load_csv_files_to_dataframe(csv_files)



