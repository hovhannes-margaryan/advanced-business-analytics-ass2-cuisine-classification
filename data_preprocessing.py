import numpy as np
import pandas as pd
from argparse import ArgumentParser


def combine(row):
    return [{image_id: row["CUISINES"]} for image_id in row["IMAGE_IDS"]]


def preprocess_dataset(input_json, input_parquet, train_parquet_path, validation_parquet_path, test_parquet_path):
    """
    :param input_json: path to input json file provided by the assignment
    :param input_parquet: path to annotated parquet file
    :param train_parquet_path: path to save train parquet
    :param validation_parquet_path: path to save validation parquet
    :param test_parquet_path: path to save test parquet
    :return:
    Saves train, validation, test parquet with columns (filtered by the 20th most common cuisines - modern french, french contemporary,
     french are combined together)
        LOCAL_PATH: local path of the image.
        LABEL: 1 if the image is food 0 otherwise.
        SCORES_SUM: probability of an image being food image.
        IMAGE_ID: id of the image.
        CUISINES: cuisine of the image
        TARGET: index of the cuisine
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

    labels = sorted(set(df_merged_filtered["CUISINES"]))
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    df_merged_filtered["TARGET"] = df_merged_filtered.apply(lambda x: label2id[x["CUISINES"]], axis=1)

    train_size = int(90 * len(df_merged_filtered) / 100)
    validation_size = int(10 * train_size / 100)

    indices = np.arange(0, len(df_merged_filtered), step=1)
    train_indices = np.random.choice(indices, train_size, replace=False)
    validation_indices = np.random.choice(train_indices, validation_size, replace=False)
    test_indices = [i for i in indices if i not in train_indices]
    train_indices = [i for i in train_indices if i not in validation_indices]

    train_df = df_merged_filtered.iloc[train_indices]
    validation_df = df_merged_filtered.iloc[validation_indices]
    test_df = df_merged_filtered.iloc[test_indices]

    train_df.reset_index(drop=True, inplace=True)
    validation_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_df.to_parquet(train_parquet_path)
    validation_df.to_parquet(validation_parquet_path)
    test_df.to_parquet(test_parquet_path)


parser = ArgumentParser()
parser.add_argument("--input_json", default="dataset.json", help="path to dataset json", type=str)
parser.add_argument("--input_parquet", default="dataset_paths.parquet", help="path to annotated parquet", type=str)
parser.add_argument("--output_parquet", default="dataset_paths.parquet", help="path to output parquet", type=str)
parser.add_argument("--train_parquet_path", default="train.parquet", help="path to train parquet", type=str)
parser.add_argument("--validation_parquet_path", default="validation.parquet", help="path to validation parquet", type=str)
parser.add_argument("--test_parquet_path", default="test.parquet", help="path to test parquet", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    preprocess_dataset(args.input_json, args.input_parquet, args.output_parquet,
                       args.train_parquet_path, args.validation_parquet_path, args.test_parquet_path)
