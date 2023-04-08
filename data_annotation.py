import tqdm
import pandas as pd
from PIL import Image
from transformers import pipeline
from argparse import ArgumentParser


def annotate_images(input_parquet: str, output_parquet: str):
    """
    :param input_parquet: path to input parquet
    :param output_parquet: path to output parquet
    :return:
    The input parquet should contain a column LOCAL_PATH (local path of the image).
    Saves output parquet with columns
            LOCAL_PATH: local path of the image.
            LABEL: 1 if the image is food 0 otherwise.
            SCORES_SUM: probability of an image being food image.
    """
    df = pd.read_parquet(args.input_parquet)
    paths = list(df["LOCAL_PATH"])
    df["LABEL"] = len(df) * ["blah"]
    df["SCORES_SUM"] = len(df) * [0.0]

    print(f"Input parquet loaded from {input_parquet}.")
    print(f"Total number of images to annotate: {len(df)}.")

    i = 0
    number_of_skipped = 0
    classifier = pipeline("image-classification", model="nateraw/food")

    print(f"Image Annotation started!")

    for path in tqdm.tqdm(paths):
        try:
            image = Image.open(path).convert("RGB")
            output = classifier(image)
        except:
            i += 1
            number_of_skipped += 1
            continue

        score_sum = sum([o["score"] for o in output])
        df["LABEL"][i] = str(not (score_sum <= 0.3))
        df["SCORES_SUM"][i] = score_sum
        i += 1

        if i % 1000 == 0:
            df.to_parquet(output_parquet)

    print(f"Image Annotation finished!")
    print(f"Output parquet saved to {output_parquet}.")
    print(f"Number of images skipped: {number_of_skipped}.")


parser = ArgumentParser()
parser.add_argument("--input_parquet", default="dataset_paths.parquet", help="path to input parquet", type=str)
parser.add_argument("--output_parquet", default="dataset_paths.parquet", help="path to output parquet", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    annotate_images(args.input_parquet, args.output_parquet)
