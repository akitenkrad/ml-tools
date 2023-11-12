from enum import Enum
from pathlib import Path

import requests

from ml_tools.utils.utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def download_from_google_drive(id: str, dst_filename: str):
    """download a shared file from Google Drive

    Args:
        id (str): _description_
        dst_filename (str): _description_
    """
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response: requests.Response = session.get(URL, params={"id": id}, stream=True, timeout=None)
    token = __get_confirm_token(response)
    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True, timeout=None)
    __save_google_drive_content(response, dst_filename)


def __get_confirm_token(response: requests.Response):
    if "virus scan warning" in response.text.lower():
        return "t"
    return None


def __save_google_drive_content(response: requests.Response, dst_filename: str):
    CHUNK_SIZE = 32768
    path = Path(dst_filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = int(len(response.content) / CHUNK_SIZE)
    with open(path, mode="wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE), total=total, desc=f"save {dst_filename} ..."):
            if chunk:
                f.write(chunk)


class GDriveObjects(Enum):
    # Fonts
    CODEM_BOLD_EXPANDED = "1x0DXiFkLz9eU2ZA1ywBG6eB6jKXIQZd7"
    CODEM_BOLD_WIDE = "1xVhJgQMSfFx2nzU--9pfOKjH_bib7bv8"
    CODEM_BOLD = "1nuPDPnNmu9TOaWh4q7A1t79534yWPOHL"
    CODEM_REGULAR_EXPANDED = "1p-MvvShOmHk_AEkAfp5u3MQDYF4WC8Jc"
    CODEM_REGULAR_WIDE = "1OC-jmcd890UCO1yLzguE22OgY8FcZPCv"
    CODEM_REGULAR = "17prO_bjpHfcBT6faTX_TGgQkAINJUsfQ"

    # MS-MARCO
    MSMARCO_TRAIN_DATA = "1Ul_5hZ1znlkmjciAVkL-R6mTvN2ZPUcF"
    MSMARCO_TRAIN_ROUGE = "10JXIrBqvWenEdVkI3b0MPeddx0VHr_w-"
    MSMARCO_DEV_DATA = "1bkIL69AjznJfdPEjgGlxORq-i_uNGgC3"
    MSMARCO_DEV_ROUGE = "11L08_GFK5lj9607or9JaZ6WR0SlxYEo4"
    MSMARCO_EXP_DATA = "1tVtIYSs7YDn6lv-1hK5WrDvS-Sj-Et5u"

    # Conference Datasets
    NeurIPS_2021 = "1JmYpSatyr2OTCPpovUDQa3PzJmz39USV"
    ANLP_2022 = "1A3Kk_tpaPPAd543j5xApco89gHAnKMCF"

    # --- JSAI
    JSAI_2023 = "1mU0ct3hjq6G16-qVKKhUd9VES5iRNALS"
    JSAI_2022 = "1-93EeiCJp5vyXatK7H6uLSIyozLeUFIL"

    # --- ACL
    ACL_2023 = "1zPVtt8PeiODN4mog0rWiZ5Zc8KQcc6Nj"
    ACL_2022 = "1-PtEnglF1SNk38H-yDMg4chAQf7KqsJe"
    ACL_2021 = "1-4QDQyhncgxD9gm7kL-d-tt_C3DSozcm"
    ACL_2020 = "1-Tr5OC6vNF6JqNXHCJzp6OP5jjRmyH9z"
    ACL_2019 = "1-2ZHSpn3D0XlW99YDwaRSTCxdFuL-ADY"
    ACL_2018 = "1-2nKJzexK0gcrwfq15VXJq3WQWa4rvMz"
    ACL_2017 = "1-6csW5aC0QjnRfpw9R4rd7EAxxgZsXeN"
    ACL_2016 = "1-MAC2MsQMJS-hEO27LhhYl32tSNyaOVA"
    ACL_2015 = "1-WuFN8fT79pywVdpKI5CCfpBgbDZkprM"
    ACL_2014 = "108SvoVmCSW1rQTtro-3g3yUX214s8nI3"
    ACL_2013 = "10BwuaR5EzpVf2Ng95f_8plIkgo8Dhs2W"
    ACL_2012 = "10F63M0ma4LPAXmNhRZyV24APVDgmNrLH"
    ACL_2011 = "10JiTpbXGQLeIN-lpXcsczXrbaxLZWSgl"
    ACL_2010 = "10Kx99fhUhS8UCmrsgeY22-gy2nTzX2jR"
    ACL_2009 = "10LFMs0yOL5izPf8qCoXLqi42hjgc8P3i"
    ACL_2008 = "10YI36bGv1gCodo4b7nEzS-LDH86ywzOB"
    ACL_2007 = "10e-jWyJEg9vJfdkdA9j-QplBcOfva5o-"
    ACL_2006 = "10eGgIMRsX_QDy7Dn69uTXxS1x-FYm0Lp"
    ACL_2005 = "10iSwVVBWd2S6tPsy4n_vWbOSZSLsCLih"
    ACL_2004 = "10mA4MGJkjW38ZKt_siwEASU_sJ4Z2X2S"
    ACL_2003 = "1-440109UsDna3DOKyWkmhGjdo4V2HrHB"
    ACL_2002 = "1-DGgtBbdRJH5wYfQzGDF4mHIfiyymBH_"
    ACL_2001 = "1-IeL_6Q8Jik_-_ttb9yghL7mDM9D7524"
    ACL_2000 = "1-QXRxWM7aNuH14nQnMyofiGhCJmufod1"
