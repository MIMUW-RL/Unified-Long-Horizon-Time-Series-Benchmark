import os

import gdown


def main():
    gdown.download_folder(
        "https://drive.google.com/drive/folders/1u3HN75OCSOl-V7VFdkDCkkQTqMGB5j7K?usp=share_link",  # noqa
        quiet=False,
        use_cookies=False,
    )
    os.system("mkdir data/mackey_glass/")
    os.system("mv mackey_glass/* data/mackey_glass/")

    gdown.download_folder(
        "https://drive.google.com/drive/folders/1xrgVk9dN0IfFUPH7pGnYo1vbuXcpoDt5?usp=share_link",  # noqa
        quiet=False,
        use_cookies=False,
    )
    os.system("mkdir data/lorenz/")
    os.system("mv lorenz/* data/lorenz/")

    gdown.download_folder(
        "https://drive.google.com/drive/folders/1EvrTrh2uvZ4RrmLluzFM1-sQk_9y-hfq?usp=share_link",  # noqa
        quiet=False,
        use_cookies=False,
    )
    os.system("mkdir data/mso/")
    os.system("mv mso/* data/mso/")

    gdown.download_folder(
        "https://drive.google.com/drive/folders/1fkWPXvizNylEPbTApOUx_hk2Eczuruy-?usp=share_link",  # noqa
        quiet=False,
        use_cookies=False,
    )
    os.system("mkdir data/sspiral/")
    os.system("mv sspiral/* data/sspiral/")



if __name__ == "__main__":
    main()
