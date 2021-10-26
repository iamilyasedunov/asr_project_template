import logging
from typing import List
import torch
from torch.nn.functional import pad

logger = logging.getLogger(__name__)


def collate_elements(dataset_entries: List[dict], key: str):
    return [entry[key] for entry in dataset_entries]


def collate_tensors(dataset_entries: List[dict], key: str, axis=0):
    return torch.cat([entry[key] for entry in dataset_entries], dim=axis)


def collate_and_pad_tensors(dataset_entries: List[dict], key: str,
                            concat_axis=0, pad_axis=-1, pad_value=0.0):
    tensors = [entry[key] for entry in dataset_entries]
    shapes = [t.shape for t in tensors]
    shapes_T = list(zip(*shapes))
    for dim_num, dim_lengths in enumerate(shapes_T):
        if dim_num == pad_axis and not all([l == dim_lengths[0] for l in dim_lengths]):
            raise Exception(f"Not all tensors are of the same len for dim {dim_num}")
    max_size = max(shapes_T[pad_axis])
    tensors_padded = []
    for i, tensor in enumerate(tensors):
        pad_right = max_size - shapes[i][pad_axis]
        tensors_padded.append(pad(tensor, (0, pad_right), mode="constant", value=pad_value))
    return torch.cat(tensors_padded, dim=concat_axis), torch.tensor(shapes_T[pad_axis], dtype=int)


def collate_fn(dataset_items: List[dict]):
    res = {
        "duration": collate_elements(dataset_items, "duration"),
        "audio_path": collate_elements(dataset_items, "audio_path"),
        "text": collate_elements(dataset_items, "text"),
        "audio": collate_elements(dataset_items, "audio")
    }

    res["spectrogram"], res["spectrogram_length"] = collate_and_pad_tensors(dataset_items, "spectrogram")

    res["text_encoded"], res["text_encoded_length"] = collate_and_pad_tensors(dataset_items,
                                                                              "text_encoded",
                                                                              0)
    # print(f" spec shape : {res['spectrogram'].shape}, spec len shape: {res['spectrogram_length'].shape}")
    return res


def __collate_fn(dataset_items: List[dict]):
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

    for field in list(dataset_items_dict_of_lists.keys()) + ["text_encoded_length", "spectrogram_length"]:
        if field in ['spectrogram', 'text_encoded', 'audio']:
            # print(f"field: {field} shape: {dataset_items_dict_of_lists[field].shape}")
            # try:
            result_batch[field] = torch.nn.utils.rnn.pad_sequence(dataset_items_dict_of_lists[field], batch_first=True)
            # except:

        elif field in ["text_encoded_length", "spectrogram_length"]:
            field_target = "spectrogram" if field == "spectrogram_length" else "text_encoded"
            result_batch[field] = torch.Tensor(
                [text_encoded_item.size(0) for text_encoded_item in dataset_items_dict_of_lists[field_target]]).type(
                torch.int64)
        else:
            result_batch[field] = dataset_items_dict_of_lists[field]

    # result_batch["spectrogram_length"] = torch.full(size=(len(dataset_items),), fill_value=result_batch["spectrogram"].shape[1])

    return result_batch
