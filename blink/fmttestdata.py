import argparse
from collections import defaultdict
from datetime import datetime
import csv
import json
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import blink.ner as NER
import blink.candidate_ranking.utils as utils

def create_uuid(kindofuniqueid:str, datetimestr:str)->str:
    return kindofuniqueid + "_" +str(datetime.fromisoformat(datetimestr)).replace(" ","T")

def sentence_dataset_tokenizer(inst:str) -> list:
    inst = inst.splitlines()
    pat = re.compile("\.\s+")
    for section in inst:
        for sentence in re.split(pat, section):
            sentence = sentence.strip()
            if len(sentence) > 0:
                yield sentence

def sentence_dataset_prep(uuid_fpath:str, test_fpath:str, *fpaths):
    """ Writes out a UUID file and a input file for SentenceDataset. 

        ***
        Assumes your files have the standard attributes "uuid", "text".
        ***
    """
    UUID = "uuid"
    TEXT = "text"
    prep = []
    for fpath in fpaths:
        logger.info("Processing {}".format(fpath))

        uuids = defaultdict(list)

        with open(fpath, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:

                instid = row[UUID]
                text = row[TEXT]

                for sentence in sentence_dataset_tokenizer(text):
                    uuids[sentence].append(instid)

        prep.append((fpath, uuids))

    with open(uuid_fpath, "w") as f:
        json.dump(prep, f)
    logger.info("Wrote out test UUID file. {}".format(uuid_fpath))

    with open(test_fpath, "w") as f:
        for _, uuids in prep:
            for sentence in uuids.keys():

                f.write(sentence + "\n")
    logger.info("Wrote out test file. {}".format(test_fpath))
    return 0


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

def run(args, logger):

    if args.dataprep:
        logger.info("Preparing test data")
        
        if not args.uuid_fpath or not args.test_fpath:
            logger.error("--uuid_fpath and --test_fpath required for output")
            return 1

        return sentence_dataset_prep(args.uuid_fpath, args.test_fpath, *args.test_files)

    return 1

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

    parser.add_argument(
        "--output_path",  help="Output filepath for predictions."
    )

    # data preparation

    parser.add_argument(
        "--dataprep", action="store_true", help="Prepare test data for NER model."
    )

    parser.add_argument(
        "--uuid_fpath", help="Filepath for UUID mapped to each sentence."
    )

    parser.add_argument(
        "--test_fpath", help="Filepath for prepared test file."
    )

    parser.add_argument(
        "--test_files", nargs="+",
        help="Filepaths for test files to be prepared."
    )

    #input_size = 5
    #output_size = 2

    #batch_size = 30
    #data_size = 100

    args = parser.parse_args()
    if not args.output_path:
        raise IOError("Output path mandatory")
    logger = utils.get_logger(args.output_path)


    if run(args, logger) != 0:
        parser.print_help()

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #rand_loader = DataLoader(dataset=SentenceDataset(input_size, data_size),
    #        batch_size=batch_size, shuffle=True)

    #ner_model = NER.get_model()

    #if torch.cuda.device_count() > 1:
    #    print("Let's use ", torch.cuda.device_count(), " GPU's!")
    #    ner_model = nn.DataParallel(ner_model)

    #ner_model.to(device)

    #for data in rand_loader:
    #    input = data.to(device)
    #    output = ner_model.predict(input)
    #    print("Outside: input size", input.size(),
    #            "output_size", output.size())

