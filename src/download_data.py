import os
import gdown


def main():
    gdown.download_folder(
        "https://drive.google.com/drive/folders/1IfAHpka3hu2kM4j6ebzAPnGpLpSMnTlf?usp=share_link",  # noqa
        quiet=False,
        use_cookies=False,
    )
    os.system("mv Final_hdf5 data")


if __name__ == "__main__":
    main()
