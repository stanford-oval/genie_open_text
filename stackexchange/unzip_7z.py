from argparse import ArgumentParser

import py7zr


def extract_7z_file(filepath, output_folder):
    with py7zr.SevenZipFile(filepath, mode='r') as z:
        z.extractall(path=output_folder)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--zip_file', type=str)
    parser.add_argument('--result_dir', type=str)

    args = parser.parse_args()
    extract_7z_file(args.zip_file, args.result_dir)
