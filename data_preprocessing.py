import pandas as pd
from argparse import ArgumentParser


def combine(row):
    return [{image_id: row["CUISINES"]} for image_id in row["IMAGE_IDS"]]


def preprocess_dataset(input_json, input_parquet, output_parquet):
    """
    :param input_json: path to input json file provided by the assignment
    :param input_parquet: path to annotated parquet file
    :param output_parquet: path to output parquet
    :return:
    Saves output parquet with columns (filtered by the 20th most common cuisines - modern french, french contemporary,
     french are combined together)
        LOCAL_PATH: local path of the image.
        LABEL: 1 if the image is food 0 otherwise.
        SCORES_SUM: probability of an image being food image.
        IMAGE_ID: id of the image.
        CUISINES: cuisine of the image
    """
    df = pd.read_json(input_json)
    df = df[["cuisines", "more_details"]]

    french = ["modern french", "french contemporary", "french"]

    df["cuisines"] = df.cuisines.apply(lambda cuisines: [cuisine["label"] for cuisine in cuisines])
    df["cuisines"] = df["cuisines"].apply(lambda cuisines: " ".join(str(cuisine).lower() for cuisine in cuisines))
    df["cuisines"] = df.apply(lambda x: "french" if x["cuisines"] in french else x["cuisines"], axis=1)

    df_count = df.groupby(df.cuisines, as_index=False).size().sort_values(by=["size"], ascending=False)[:20]

    df["more_details"] = df.more_details.apply(lambda images: [image["image_id"] for image in images["full_images"]])
    df.rename(columns={"cuisines": "CUISINES", "more_details": "IMAGE_IDS"}, inplace=True)

    ids_and_cuisines = [item for sublist in list(df.apply(combine, axis=1)) for item in sublist]

    df_images = pd.DataFrame(
        zip([list(image.keys())[0] for image in ids_and_cuisines],
            [list(image.values())[0] for image in ids_and_cuisines]),
        columns=["IMAGE_ID", "CUISINES"]
    )

    df_local_paths = pd.read_parquet(input_parquet)
    df_local_paths["IMAGE_ID"] = df_local_paths.LOCAL_PATH.apply(lambda x: x.split("/")[-1].split(".")[0])

    df_merged = df_local_paths.merge(df_images, on="IMAGE_ID")
    df_merged_filtered = df_merged[df_merged.CUISINES.isin(list(df_count.cuisines))]
    df_merged_filtered = df_merged_filtered[df_merged_filtered.LABEL == "True"]
    df_merged_filtered.reset_index(drop=True, inplace=True)
    df_merged_filtered.to_parquet(output_parquet)


parser = ArgumentParser()
parser.add_argument("--input_json", default="dataset.json", help="path to dataset json", type=str)
parser.add_argument("--input_parquet", default="dataset_paths.parquet", help="path to annotated parquet", type=str)
parser.add_argument("--output_parquet", default="dataset_paths.parquet", help="path to output parquet", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    preprocess_dataset(args.input_json, args.input_parquet, args.output_parquet)