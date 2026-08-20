import argparse
from pathlib import Path


def find_h5_files(root_dir, name_contains=None):
    root_path = Path(root_dir)
    files = root_path.rglob('*.h5')
    if name_contains:
        files = [path for path in files if name_contains in path.name]
    return sorted(str(path) for path in files)


def save_list(file_list, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in file_list:
            f.write(item + '\n')


def generate_txt_from_folder(root_dir, output_txt, name_contains=None):
    file_list = find_h5_files(root_dir, name_contains=name_contains)
    if not file_list:
        detail = f" matching {name_contains!r}" if name_contains else ""
        raise FileNotFoundError(f'No .h5 files{detail} found under: {root_dir}')
    Path(output_txt).parent.mkdir(parents=True, exist_ok=True)
    save_list(file_list, output_txt)
    return file_list




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate a list of .h5 files from a folder')
    parser.add_argument('--root_dir', type=str, required=True, help='Path to the root directory containing .h5 files')
    parser.add_argument('--output_txt', type=str, required=True, help='Path to the output text file')
    parser.add_argument(
        '--name_contains',
        type=str,
        default=None,
        help='Only include H5 files whose filename contains this text.',
    )
    args = parser.parse_args()

    file_list = generate_txt_from_folder(
        args.root_dir,
        args.output_txt,
        name_contains=args.name_contains,
    )
    print(f'Saved {len(file_list)} .h5 files to {args.output_txt}')


'''
python script/genetate_h5_txt.py \
  --root_dir /path/to/processed/PURE \
  --output_txt /path/to/output/pure_h5.txt

'''
