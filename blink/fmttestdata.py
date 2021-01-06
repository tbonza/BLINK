import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
import os
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

def sentence_dataset_prep(workdir:str, uuid_fpath:str, test_fpath:str, *fpaths):
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

    with open(os.path.join(workdir, uuid_fpath), "w") as f:
        json.dump(prep, f)
    logger.info("Wrote out test UUID file. {}".format(os.path.join(workdir, uuid_fpath)))

    with open(os.path.join(workdir, test_fpath), "w") as f:
        for _, uuids in prep:
            for sentence in uuids.keys():

                f.write(sentence + "\n")
    logger.info("Wrote out test file. {}".format(os.path.join(workdir, test_fpath)))
    return 0

class SentenceDataset(Dataset):
    """ Sentence Dataset

    Input file should be a newline delimited file with one
    sentence per line. Each line should be unique so we
    don't run the same prediction multiple times.
    """

    def __init__(self, fpath:str):
        with open(fpath,"r") as f:
            self.data = f.readlines()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.data[idx]


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

def batch_ner(args, logger):

    batch_size = args.batch_size
    data_path = os.path.join(args.workdir, args.test_fpath)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_loader = DataLoader(dataset=SentenceDataset(fpath=data_path),
            batch_size=batch_size, shuffle=False)

    ner_model = NER.get_model()

    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs.".format(torch.cuda.device_count())
        ner_model = nn.DataParallel(ner_model)

    ner_model.to(device)

    for data in data_loader:
        input = data.to(device)
        output = ner_model.predict(input)
        logger.info("Input size {}, output size {}".\
                format(input.size(), output.size()))
    return 0

def run(args, logger):

    if not os.path.exists(args.workdir):
        os.makedirs(args.workdir)

    if args.dataprep:
        logger.info("Preparing test data")
        
        if not args.uuid_fpath or not args.test_fpath:
            logger.error("--uuid_fpath and --test_fpath required for output")
            return 1

        return sentence_dataset_prep(args.workdir, args.uuid_fpath, args.test_fpath, 
                *args.test_files)

    elif args.nermodel:
        logger.info("Running batch predictions for NER model")
        return batch_ner(args, logger)

    return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--workdir",  help="Working directory for test and log files."
    )

    # NER model batch predictions 

    parser.add_argument(
        "--nermodel", action="store_true", help="Select NER model for batch preditions."
    )

    parser.add_argument(
        "--batch_size", type=int, help="Size of each batch processed."
    )

    parser.add_argument(
        "--test_fpath", default="test_file.txt",
        help="Filepath for prepared test file."
    )

    # data preparation

    parser.add_argument(
        "--dataprep", action="store_true", help="Prepare test data for NER model."
    )

    parser.add_argument(
        "--uuid_fpath", default="test_uuid.json",
        help="Filepath for UUID mapped to each sentence."
    )

    parser.add_argument(
        "--test_files", nargs="+",
        help="Filepaths for test files to be prepared."
    )

    args = parser.parse_args()
    if not args.workdir:
        raise IOError("Working directory path mandatory")
    logger = utils.get_logger(args.workdir)

    if run(args, logger) != 0:
        parser.print_help()
