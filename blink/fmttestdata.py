import argparse
from collections import defaultdict
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import blink.ner as NER


def sentence_dataset_tokenizer(inst:str) -> list:
    pass

def sentence_dataset_prep(*args):
    """ Writes out a UUID file and a input file for SentenceDataset. """
    prep = []
    for inst in args:

        uuids = defaultdict(list)

        uuid = inst["uuid"]
        fpath = inst["fpath"]
        text_id = inst["text_id"]

        with open(fpath, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:

                instid = row[uuid]
                text = row[text_id]

                for sentence in sentence_dataset_tokenizer(text):
                    uuids[sentence].append(uuid)


        prep.append((fpath, uuids))

    return prep

class SentenceDataset(Dataset):
    """ Sentence Dataset

    Input file should be a newline delimited file with one
    sentence per line. Each line should be unique so we
    don't run the same prediction multiple times.
    """

    def __init__(self, fpath:str):
        pass
    
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        pass

def _annotate(ner_model, input_sentences):
    ner_output_data = ner_model.predict(input_sentences)
    sentences = ner_output_data["sentences"]
    mentions = ner_output_data["mentions"]
    samples = []
    for mention in mentions:
        record = {}
        record["label"] = "unknown"
        record["label_id"] = -1
        # LOWERCASE EVERYTHING !
        record["context_left"] = sentences[mention["sent_idx"]][
            : mention["start_pos"]
        ].lower()
        record["context_right"] = sentences[mention["sent_idx"]][
            mention["end_pos"] :
        ].lower()
        record["mention"] = mention["text"].lower()
        record["start_pos"] = int(mention["start_pos"])
        record["end_pos"] = int(mention["end_pos"])
        record["sent_idx"] = mention["sent_idx"]
        samples.append(record)
    return samples

def run(args, logger, model):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # test data

    parser.add_argument(
        "--test_input_size", "-i", action="store_true", help="Size of test input."
    )

    parser.add_argument(
        "--test_output_size", "-o", action="store_true", help="Size of test output."
    )

    # batch 

    parser.add_argument(
        "--batch_size", action="store_true", help="Size of each batch processed."
    )

    parser.add_argument(
        "--output_size", action="store_true", help="Size of total amount of data."
    )


    #input_size = 5
    #output_size = 2

    #batch_size = 30
    #data_size = 100

    args = parser.parse_args()
    run(args, logger, *models)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rand_loader = DataLoader(dataset=SentenceDataset(input_size, data_size),
            batch_size=batch_size, shuffle=True)

    ner_model = NER.get_model()

    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), " GPU's!")
        ner_model = nn.DataParallel(ner_model)

    ner_model.to(device)

    for data in rand_loader:
        input = data.to(device)
        output = ner_model.predict(input)
        print("Outside: input size", input.size(),
                "output_size", output.size())

