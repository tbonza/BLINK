import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import blink.ner as NER

class SentenceDataset(Dataset):
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

def run():
    pass

if __name__ == "__main__":

    # Parameters and DataLoaders

    input_size = 5
    output_size = 2

    batch_size = 30
    data_size = 100

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

