import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
import os
import re
from multiprocessing import cpu_count, Pool, current_process

import blink.ner as NER
from flair.models import SequenceTagger
import blink.candidate_ranking.utils as utils

def _create_uuid(kindofuniqueid:str, datetimestr:str)->str:
    return kindofuniqueid + "_" +str(datetime.fromisoformat(datetimestr)).replace(" ","T")

def _map_attributes(attrs:list, logger)->list:
    """ Given an attribute mapping, convert csv row to standard format. 

    Example: "uuid:pplcd,lastdate[datetime]; text:discussion_summary" becomes
        { "uuid": [("str", "pplcd"), ("datetime", "lastdate")], "text": ["discussion_summary"] }

    Datetime is only supported if it's in ISO format.
    """
    mappings = []
    template = {
        "uuid": "",
        "text": "",
    }
    for attrstr in attrs:

        am = template.copy()
        for segment in attrstr.split(";"):
            if "uuid:" in segment:
                u = []
                _, uuids = segment.split(":")
                for inst in uuids.split(","):

                    if "[datetime]" in inst:
                        u.append(("datetime", inst.replace("[datetime]","").strip()))

                    else:
                        u.append(("str", inst.strip()))

                am["uuid"] = u

            elif "text:" in segment:
                t = []
                _, texts = segment.split(":")
                for inst in texts.split(","):
                    t.append(inst)

                am["text"] = t

            else:
                logger.error("Attribute type not supported: {}".format(segment))
                return {}, 1

        mappings.append(am)
    
    return mappings, 0

def _verify_fieldnames(fieldnames:set, attrmaps:list) -> dict:
    for attrmap in attrmaps:
        cols = []
        for k,v in attrmap.items():
            if k == "uuid":
                for inst in v:
                    cols.append(inst[1])
            else:
                cols.extend(v)
        if len(set(cols) - fieldnames) == 0:
            return attrmap, 0

    return {}, 1

def _sentence_dataset_tokenizer(inst:str) -> list:
    inst = inst.splitlines()
    pat = re.compile("\.\s+")
    for section in inst:
        for sentence in re.split(pat, section):
            sentence = sentence.strip()
            if len(sentence) > 0:
                yield sentence

def _isodate_tostr(dt:str) -> str:
    return str(datetime.fromisoformat(dt)).replace(" ","T")

def _refmt_row(row:dict, attrmap:dict, logger) -> dict:
    uuid = ""
    text = ""
    keyz = { i[1]:i[0] for i in attrmap["uuid"] }
    txtz = set(attrmap["text"])
    for k in row.keys():

        if k in keyz:
            
            if keyz[k] == "str":
                uuid += (row[k] + "_")

            elif keyz[k] == "datetime":
                uuid += (_isodate_tostr(row[k]) + "_")

            else:
                logger.error("uuid type not supported. {}".format(k))
                return {}, 1

        elif k in txtz:
            text += (row[k] + "\n")

    uuid = uuid[:-1] # remove trailing underscore
    return {"uuid": uuid, "text": text}, 0

def _sentence_dataset_prep(workdir:str, uuid_fpath:str, test_fpath:str, attrmaps:list, *fpaths):
    """ Writes out a UUID file and a input file for SentenceDataset. 

        ***
        Assumes your files have the standard attributes "uuid", "text".
        ***
    """
    UUID = "uuid"
    TEXT = "text"
    prep = []
    amap = False

    if len(attrmaps) > 0:
        attrmaps, status = _map_attributes(attrmaps, logger)
        if status != 0:
            return status
        amap = True

    for fpath in fpaths:
        logger.info("Processing {}".format(fpath))

        uuids = defaultdict(list)

        with open(fpath, "r") as f:
            reader = csv.DictReader(f)

            if amap:
            
                fieldnames = set(reader.fieldnames)
                attrmap, status = _verify_fieldnames(fieldnames, attrmaps)
                if status != 0:
                    return status

                for row in reader:
                    
                    row, status = _refmt_row(row, attrmap, logger)

                    if status == 0:
                        for sentence in _sentence_dataset_tokenizer(row[TEXT]):
                            uuids[sentence].append(row[UUID])
                    else:
                        logger.warning("Unable to convert row to standard attributes. {}".format(row))
            else:

                for row in reader:

                    for sentence in _sentence_dataset_tokenizer(row[TEXT]):
                        uuids[sentence].append(row[UUID])

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

def _pannotate(args):
    ner_model, input_sentences, logger = args
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
    current = current_process()
    logger.info("Processed batch size {} using {}".\
            format(len(input_sentences), current.name))
    return samples

def _batch_ner(args, logger):

    batch_size = args.batch_size
    data_path = os.path.join(args.workdir, args.test_fpath)
    ner_model = NER.get_model()

    with open(data_path,"r") as f:
        test_data = f.readlines()

    logger.info("Read test data with {} instances.".format(len(test_data)))
    test_batches = [ (ner_model, test_data[i:i+batch_size], logger) 
            for i in range(0,len(test_data),batch_size) ]
    del test_data

    logger.info("Starting NER predictions")
    results = []
    with Pool(cpu_count()) as p:
        results = p.map(_pannotate, test_batches)
    logger.info("Finished NER predictions")

    with open(os.path.join(args.workdir, args.predict_fpath), "w") as f:
        json.dump(results, f)

    return 0

def run(args, logger):

    if not os.path.exists(args.workdir):
        os.makedirs(args.workdir)

    if args.dataprep:
        logger.info("Preparing test data")
        
        if not args.uuid_fpath or not args.test_fpath:
            logger.error("--uuid_fpath and --test_fpath required for output")
            return 1

        return _sentence_dataset_prep(args.workdir, args.uuid_fpath, args.test_fpath, 
                [*args.attrmaps], *args.test_files)

    elif args.nermodel:
        logger.info("Running batch predictions for NER model")
        return _batch_ner(args, logger)

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

    parser.add_argument(
        "--predict_fpath", default="test_predict.json",
        help="Filepath for model batch predictions."
    )

    # data preparation

    parser.add_argument(
        "--dataprep", action="store_true", help="Prepare test data for NER model."
    )

    parser.add_argument(
        "--attrmaps", nargs="+",
        help="Map attributes to standard attributes for test files."
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
