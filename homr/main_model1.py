import argparse
import glob
import os
import sys

import cv2
import numpy as np

from homr import color_adjust, download_utils
from homr.autocrop import autocrop
from homr.debug import Debug
from homr.model import InputPredictions
from homr.noise_filtering import filter_predictions
from homr.resize import resize_image
from homr.segmentation.config import segnet_path, unet_path
from homr.segmentation.segmentation import segmentation
from homr.simple_logging import eprint
from homr.transformer.configs import default_config
from homr.type_definitions import NDArray
from homr.staff_detection import make_lines_stronger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_predictions(
    original: NDArray, preprocessed: NDArray, img_path: str, save_cache: bool
) -> InputPredictions:
    result = segmentation(preprocessed, img_path, use_cache=save_cache)
    original_image = cv2.resize(original, (result.staff.shape[1], result.staff.shape[0]))
    preprocessed_image = cv2.resize(preprocessed, (result.staff.shape[1], result.staff.shape[0]))
    return InputPredictions(
        original=original_image,
        preprocessed=preprocessed_image,
        notehead=result.notehead.astype(np.uint8),
        symbols=result.symbols.astype(np.uint8),
        staff=result.staff.astype(np.uint8),
        clefs_keys=result.clefs_keys.astype(np.uint8),
        stems_rest=result.stems_rests.astype(np.uint8),
    )


def load_and_preprocess_predictions(
    image_path: str, enable_debug: bool, enable_cache: bool
) -> tuple[InputPredictions, Debug]:
    image = cv2.imread(image_path)
    image = autocrop(image)
    image = resize_image(image)
    preprocessed, _background = color_adjust.color_adjust(image, 40)
    predictions = get_predictions(image, preprocessed, image_path, enable_cache)
    debug = Debug(predictions.original, image_path, enable_debug)
    debug.write_image("color_adjust", preprocessed)

    predictions = filter_predictions(predictions, debug)

    predictions.staff = make_lines_stronger(predictions.staff, (1, 2))
    debug.write_threshold_image("staff", predictions.staff)
    debug.write_threshold_image("symbols", predictions.symbols)
    debug.write_threshold_image("stems_rest", predictions.stems_rest)
    debug.write_threshold_image("notehead", predictions.notehead)
    debug.write_threshold_image("clefs_keys", predictions.clefs_keys)
    return predictions, debug


def process_image(  # noqa: PLR0915
    image_path: str, enable_debug: bool, enable_cache: bool
) -> tuple[str, str, str]:
    eprint("Processing " + image_path)
    predictions, debug = load_and_preprocess_predictions(image_path, enable_debug, enable_cache)
    return predictions, debug

def get_all_image_files_in_folder(folder: str) -> list[str]:
    image_files = []
    for ext in ["png", "jpg", "jpeg"]:
        image_files.extend(glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True))
    without_teasers = [
        img
        for img in image_files
        if "_teaser" not in img
        and "_debug" not in img
        and "_staff" not in img
        and "_tesseract" not in img
    ]
    return sorted(without_teasers)


def download_weights() -> None:
    base_url = "https://github.com/liebharc/homr/releases/download/checkpoints/"
    models = [segnet_path, unet_path, default_config.filepaths.checkpoint]
    missing_models = [model for model in models if not os.path.exists(model)]
    if len(missing_models) == 0:
        return

    eprint("Downloading", len(missing_models), "models - this is only required once")
    for model in missing_models:
        if not os.path.exists(model) or True:
            base_name = os.path.basename(model).split(".")[0]
            eprint(f"Downloading {base_name}")
            try:
                zip_name = base_name + ".zip"
                download_url = base_url + zip_name
                downloaded_zip = os.path.join(os.path.dirname(model), zip_name)
                download_utils.download_file(download_url, downloaded_zip)

                destination_dir = os.path.dirname(model)
                download_utils.unzip_file(downloaded_zip, destination_dir)
            finally:
                if os.path.exists(downloaded_zip):
                    os.remove(downloaded_zip)


def main_model1(imagePath='bach1001_2.png', finit=False, fdebug=False, fcache=False):
    download_weights()
    if finit:
        eprint("Init finished")
        return

    if not imagePath:
        eprint("No image provided")
        sys.exit(1)
    elif os.path.isfile(imagePath):
        predictions, debug = process_image(imagePath, fdebug, fcache)
        return predictions, debug
    elif os.path.isdir(imagePath):
        print(f'{imagePath} is a directory, not an image')
    else:
        raise ValueError(f"{imagePath} is not a valid file or directory")


if __name__ == "__main__":
    main_model1()
