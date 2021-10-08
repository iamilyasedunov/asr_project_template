import logging
from typing import List
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    def preprocess_dataset_item(dataset_item, field: str):
        if field == 'spectrogram':
            return dataset_item.squeeze().transpose(0, 1)
        elif field in ['text_encoded', 'audio']:
            return dataset_item.squeeze()
        return dataset_item

    result_batch = {}

    dataset_items_dict_of_lists = {
        field: [preprocess_dataset_item(dataset_item[field], field) for dataset_item in dataset_items] for field in
        dataset_items[0]}

    for field in list(dataset_items_dict_of_lists.keys()) + ["text_encoded_length"]:
        if field in ['spectrogram', 'text_encoded', 'audio']:
            result_batch[field] = torch.nn.utils.rnn.pad_sequence(dataset_items_dict_of_lists[field], batch_first=True)
        elif field == "text_encoded_length":
            result_batch[field] = torch.Tensor(
                [text_encoded_item.size(0) for text_encoded_item in dataset_items_dict_of_lists["text_encoded"]]).type(torch.int64)
        else:
            result_batch[field] = dataset_items_dict_of_lists[field]

    result_batch["spectrogram_length"] = torch.full(size=(len(dataset_items),), fill_value=result_batch["spectrogram"].shape[1])

    return result_batch
