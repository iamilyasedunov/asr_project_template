import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import hw_asr.model as module_model
from hw_asr.datasets.utils import get_dataloaders
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.trainer import Trainer
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.parse_config import ConfigParser
from hw_asr.metric.utils import calc_cer
from hw_asr.metric.utils import calc_wer

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


# saved/models/default_config/1014_090310/checkpoint-epoch30.pth
# batch = 20, CER (argmax): 0.027
# https://wandb.ai/iamilyasedunov/asr_project/runs/5w4if8ao?workspace=user-iamilyasedunov
# fresh-river-13

# denim-moon RNNmodel

# drawn-smoke-50 0.0112

def main(config, out_file):
    logger = config.get_logger("test")

    # text_encoder
    text_encoder = CTCCharTextEncoder.get_simple_alphabet()

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=len(text_encoder))
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    results = []
    cers = {"argmax": [], "bs": []}
    wers = {"argmax": [], "bs": []}
    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(dataloaders["test"])):
            batch = Trainer.move_batch_to_device(batch, device)
            output = model(**batch)
            if type(output) is dict:
                batch.update(output)
            else:
                batch["logits"] = output
            batch["log_probs"] = torch.log_softmax(batch["logits"], dim=-1)
            batch["log_probs_length"] = model.transform_input_lengths(
                batch["spectrogram_length"]
            )
            batch["probs"] = batch["log_probs"].exp().cpu()
            batch["argmax"] = batch["probs"].argmax(-1)
            batch["bs_predictions"] = text_encoder.ctc_beam_search(batch["log_probs"].cpu())
            for i in range(len(batch["text"])):
                argmax = batch["argmax"][i]
                argmax = argmax[:int(batch["log_probs_length"][i])]
                pred_text_argmax = text_encoder.ctc_decode(argmax.numpy())
                results.append(
                    {
                        "ground_trurh": batch["text"][i],
                        "pred_text_argmax": pred_text_argmax,
                        "pred_text_beam_search": batch["bs_predictions"][i],
                        "wer_argmax": calc_wer(batch["text"][i], pred_text_argmax),
                        "cer_argmax": calc_cer(batch["text"][i], pred_text_argmax),
                        "wer_bs": calc_wer(batch["text"][i], batch["bs_predictions"][i]),
                        "cer_bs": calc_cer(batch["text"][i], batch["bs_predictions"][i]),
                    }
                )
    results_dict_of_lists = {
        field: [result[field] for result in results] for field in results[0]
    }
    print(f"WER (argmax): {sum(results_dict_of_lists['wer_argmax']) / len( batch['text'])}")
    print(f"CER (argmax): {sum(results_dict_of_lists['cer_argmax']) / len( batch['text'])}")
    print(f"WER (beam s): {sum(results_dict_of_lists['wer_bs']) / len( batch['text'])}")
    print(f"CER (beam s): {sum(results_dict_of_lists['cer_bs']) / len( batch['text'])}")

    with Path(out_file).open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="hw_asr/config_test",
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=5,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "audio_dir": str(test_data_folder / "audio"),
                            "transcription_dir": str(
                                test_data_folder / "transcriptions"
                            ),
                        },
                    }
                ],
            }
        }

    assert config.config.get("data", {}).get("test", None) is not None
    config["data"]["test"]["batch_size"] = args.batch_size
    config["data"]["test"]["n_jobs"] = args.jobs

    main(config, args.output)
