import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "kaggle_path", help="Path to kaggle.json file.", type=str, required=True
    )
    parser.add_argument(
        "--username", help="Your Kaggle username.", type=str, required=True
    )
    parser.add_argument("--key", help="Your kaggle key.", type=str, required=True)

    return parser


parser = get_parser()
args = parser.parse_args()

with open(args.kaggle_path, "w+") as f:
    f.write(f'{{"username":"{args.username}","key":"{args.key}"}}')
    # Put your kaggle username & key here
