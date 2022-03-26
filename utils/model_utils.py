import os
from gdown import download as drive_download


google_drive_paths = {'afhqwild.pt': 'https://drive.google.com/uc?id=14OnzO4QWaAytKXVqcfWo_o2MzoR4ygnr',
                'afhqdog.pt': 'https://drive.google.com/uc?id=16v6jPtKVlvq8rg2Sdi3-R9qZEVDgvvEA',
                'afhqcat.pt': 'https://drive.google.com/uc?id=1HXLER5R3EMI8DSYDBZafoqpX4EtyOf2R',
                'ffhq.pt': 'https://drive.google.com/uc?id=1AT6bNR2ppK8f2ETL_evT27f3R_oyWNHS',
                'metfaces.pt': 'https://drive.google.com/uc?id=16wM2PwVWzaMsRgPExvRGsq6BWw_muKbf',
                'shape_predictor_68_face_landmarks.dat': 'https://drive.google.com/uc?id=17kwWXLN9fA6acrBWqfuQCBdcc1ULmBc9'
}

def download_weight(weight_path):
    if not os.path.isfile(weight_path) and (
            os.path.basename(weight_path) in google_drive_paths
    ):

        gdrive_url = google_drive_paths[os.path.basename(weight_path)]
        try:
            # drive_download(gdrive_url, weight_path, fuzzy=True)
            drive_download(gdrive_url, weight_path, quiet=False)
        except ModuleNotFoundError:
            print(
                "gdown module not found.",
                "pip3 install gdown or, manually download the checkpoint file:",
                gdrive_url
            )

