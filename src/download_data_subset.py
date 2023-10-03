import os

import gdown


def main():
    gdown.download_folder(
        "https://drive.google.com/drive/folders/172QgaXtOIPfPiI9b_anLQuuDRR-ZMBWj?usp=share_link",  # noqa
        quiet=False,
        use_cookies=False,
    )
    os.system("mkdir data/cheetah_medium_deter/")
    os.system("mv cheetah_medium_deter/* data/cheetah_medium_deter/")

    gdown.download_folder(
        "https://drive.google.com/drive/folders/1n6D3krCd6rDvQlOvb-z317LGl9qAZsB7?usp=share_link",  # noqa
        quiet=False,
        use_cookies=False,
    )
    os.system("mkdir data/hopper_medium_deter/")
    os.system("mv hopper_medium_deter/* data/hopper_medium_deter/")

    gdown.download_folder(
        "https://drive.google.com/drive/folders/16idSzUl35L33mFzzrx3MvQ2Yxkn9nnln?usp=share_link",  # noqa
        quiet=False,
        use_cookies=False,
    )
    os.system("mkdir data/walker_medium_deter/")
    os.system("mv walker_medium_deter/* data/walker_medium_deter/")


if __name__ == "__main__":
    main()
