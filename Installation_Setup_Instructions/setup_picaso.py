import warnings
warnings.filterwarnings('ignore')

import picaso.data as data
data.get_data()

### Set these inputs ###
# ck_tables
# by-molecule
# yes

# import requests
# import os

# URLS = {
#     'H2S_1460.npy':'https://zenodo.org/records/10895826/files/H2S_1460.npy',
#     'MgH_1460.npy':'https://zenodo.org/records/10895826/files/MgH_1460.npy',
#     'O2_1460.npy':'https://zenodo.org/records/10895826/files/O2_1460.npy',
#     'FeH_1460.npy':'https://zenodo.org/records/10895826/files/FeH_1460.npy',
#     'TiO_1460.npy':'https://zenodo.org/records/10895826/files/TiO_1460.npy',
#     'SO2_1460.npy':'https://zenodo.org/records/10895826/files/SO2_1460.npy',
#     'Fe_1460.npy':'https://zenodo.org/records/10895826/files/Fe_1460.npy',
#     'C2H4_1460.np':'https://zenodo.org/records/10895826/files/C2H4_1460.npy',
#     'OCS_1460.npy':'https://zenodo.org/records/10895826/files/OCS_1460.npy',
#     'C2H6_1460.npy':'https://zenodo.org/records/10895826/files/C2H6_1460.npy',
#     'SiO_1460.np':'https://zenodo.org/records/10895826/files/SiO_1460.npy',
#     'C2H2_1460.npy':'https://zenodo.org/records/10895826/files/C2H2_1460.npy',
#     'LiCl_1460.npy':'https://zenodo.org/records/10895826/files/LiCl_1460.npy',
#     'CO2_1460.npy':'https://zenodo.org/records/10895826/files/CO2_1460.npy',
#     'CrH_1460.npy':'https://zenodo.org/records/10895826/files/CrH_1460.npy',
#     'Na_1460.npy':'https://zenodo.org/records/10895826/files/Na_1460.npy',
#     'Rb_1460.npy':'https://zenodo.org/records/10895826/files/Rb_1460.npy',
#     'H3+_1460.npy':'https://zenodo.org/records/10895826/files/H3+_1460.npy',
#     'O3_1460.npy':'https://zenodo.org/records/10895826/files/O3_1460.npy',
#     'H2O_1460.npy':'https://zenodo.org/records/10895826/files/H2O_1460.npy',
#     'H2_1460.npy':'https://zenodo.org/records/10895826/files/H2_1460.npy',
#     'VO_1460.npy':'https://zenodo.org/records/10895826/files/VO_1460.npy',
#     'CO_1460.npy':'https://zenodo.org/records/10895826/files/CO_1460.npy',
#     'LiF_1460.npy':'https://zenodo.org/records/10895826/files/LiF_1460.npy',
#     'N2_1460.npy':'https://zenodo.org/records/10895826/files/N2_1460.npy',
#     'CaH_1460.npy':'https://zenodo.org/records/10895826/files/CaH_1460.npy',
#     'LiH_1460.npy':'https://zenodo.org/records/10895826/files/LiH_1460.npy',
#     'K_1460.npy':'https://zenodo.org/records/10895826/files/K_1460.npy',
#     'CH4_1460.npy':'https://zenodo.org/records/10895826/files/CH4_1460.npy',
#     'Li_1460.npy':'https://zenodo.org/records/10895826/files/Li_1460.npy',
#     'HCN_1460.npy':'https://zenodo.org/records/10895826/files/HCN_1460.npy',
#     'TiH_1460.np':'https://zenodo.org/records/10895826/files/TiH_1460.npy',
#     'Cs_1460.npy':'https://zenodo.org/records/10895826/files/Cs_1460.npy',
#     'NH3_1460.npy':'https://zenodo.org/records/10895826/files/NH3_1460.npy',
#     'PH3_1460.npy':'https://zenodo.org/records/10895826/files/PH3_1460.npy',
#     'AlH_1460.npy':'https://zenodo.org/records/10895826/files/AlH_1460.npy'
# }

# def download_file(url, local_filename):
#     """Downloads a file from a URL and saves it locally."""
#     try:
#         # Send a GET request to the URL in binary format
#         with requests.get(url, stream=True) as r:
#             r.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            
#             # Open the local file in binary write mode ('wb') and write the content
#             with open(local_filename, 'wb') as f:
#                 # Write data in chunks to handle large files efficiently
#                 for chunk in r.iter_content(chunk_size=8192): 
#                     f.write(chunk)
#         print(f"File '{local_filename}' downloaded successfully.")
#     except requests.exceptions.RequestException as e:
#         print(f"An error occurred: {e}")

# def main():
#     for key in URLS:
#         url = URLS[key]
#         download_file(url, os.path.join('picasofiles/reference/opacities/resortrebin/',key))
#         break

# if __name__ == '__main__':
#     main()
