import argparse
import glob
import os
import sys

import cv2
import numpy as np

from homr import color_adjust, download_utils
from homr.accidental_detection import add_accidentals_to_staffs
from homr.accidental_rules import maintain_accidentals
from homr.autocrop import autocrop
from homr.bar_line_detection import add_bar_lines_to_staffs, detect_bar_lines
from homr.bounding_boxes import (
    BoundingEllipse,
    RotatedBoundingBox,
    create_bounding_ellipses,
    create_rotated_bounding_boxes,
)
from homr.brace_dot_detection import (
    find_braces_brackets_and_grand_staff_lines,
    prepare_brace_dot_image,
)
from homr.debug import Debug
from homr.model import InputPredictions
from homr.noise_filtering import filter_predictions
from homr.note_detection import add_notes_to_staffs, combine_noteheads_with_stems
from homr.resize import resize_image
from homr.rest_detection import add_rests_to_staffs
from homr.segmentation.config import segnet_path, unet_path
from homr.segmentation.segmentation import segmentation
from homr.simple_logging import eprint
from homr.staff_detection import break_wide_fragments, detect_staff, make_lines_stronger
from homr.staff_parsing import parse_staffs
from homr.title_detection import detect_title
from homr.transformer.configs import default_config
from homr.type_definitions import NDArray
from homr.xml_generator import generate_xml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class PredictedSymbols:
    def __init__(
        self,
        noteheads: list[BoundingEllipse],
        staff_fragments: list[RotatedBoundingBox],
        clefs_keys: list[RotatedBoundingBox],
        accidentals: list[RotatedBoundingBox],
        stems_rest: list[RotatedBoundingBox],
        bar_lines: list[RotatedBoundingBox],
    ) -> None:
        self.noteheads = noteheads
        self.staff_fragments = staff_fragments
        self.clefs_keys = clefs_keys
        self.accidentals = accidentals
        self.stems_rest = stems_rest
        self.bar_lines = bar_lines


def replace_extension(path: str, new_extension: str) -> str:
    return (
        path.replace(".png", new_extension)
        .replace(".jpg", new_extension)
        .replace(".jpeg", new_extension)
    )


def predict_symbols(debug: Debug, predictions: InputPredictions) -> PredictedSymbols:
    eprint("Creating bounds for noteheads")
    noteheads = create_bounding_ellipses(predictions.notehead, min_size=(4, 4))
    eprint("Creating bounds for staff_fragments")
    staff_fragments = create_rotated_bounding_boxes(
        predictions.staff, skip_merging=True, min_size=(5, 1), max_size=(10000, 100)
    )

    eprint("Creating bounds for clefs_keys")
    clefs_keys = create_rotated_bounding_boxes(
        predictions.clefs_keys, min_size=(20, 40), max_size=(1000, 1000)
    )
    eprint("Creating bounds for accidentals")
    accidentals = create_rotated_bounding_boxes(
        predictions.clefs_keys, min_size=(5, 5), max_size=(100, 100)
    )
    eprint("Creating bounds for stems_rest")
    stems_rest = create_rotated_bounding_boxes(predictions.stems_rest)
    eprint("Creating bounds for bar_lines")
    bar_line_img = predictions.stems_rest
    debug.write_threshold_image("bar_line_img", bar_line_img)
    bar_lines = create_rotated_bounding_boxes(bar_line_img, skip_merging=True, min_size=(1, 5))

    return PredictedSymbols(
        noteheads, staff_fragments, clefs_keys, accidentals, stems_rest, bar_lines
    )


def process_image(  # noqa: PLR0915
    image_path: str, enable_debug: bool, enable_cache: bool, predictions: InputPredictions, debug: Debug
) -> tuple[str, str, str]:
    eprint("Processing " + image_path)
    xml_file = replace_extension(image_path, ".musicxml")
    try:
        eprint("Loaded segmentation")
        symbols = predict_symbols(debug, predictions)
        eprint("Predicted symbols")

        symbols.staff_fragments = break_wide_fragments(symbols.staff_fragments)
        debug.write_bounding_boxes("staff_fragments", symbols.staff_fragments)
        eprint("Found " + str(len(symbols.staff_fragments)) + " staff line fragments")

        noteheads_with_stems, likely_bar_or_rests_lines = combine_noteheads_with_stems(
            symbols.noteheads, symbols.stems_rest
        )
        debug.write_bounding_boxes_alternating_colors("notehead_with_stems", noteheads_with_stems)
        eprint("Found " + str(len(noteheads_with_stems)) + " noteheads")
        if len(noteheads_with_stems) == 0:
            raise Exception("No noteheads found")

        average_note_head_height = float(
            np.mean([notehead.notehead.size[1] for notehead in noteheads_with_stems])
        )
        eprint("Average note head height: " + str(average_note_head_height))

        all_noteheads = [notehead.notehead for notehead in noteheads_with_stems]
        all_stems = [note.stem for note in noteheads_with_stems if note.stem is not None]
        bar_lines_or_rests = [
            line
            for line in symbols.bar_lines
            if not line.is_overlapping_with_any(all_noteheads)
            and not line.is_overlapping_with_any(all_stems)
        ]
        bar_line_boxes = detect_bar_lines(bar_lines_or_rests, average_note_head_height)
        debug.write_bounding_boxes_alternating_colors("bar_lines", bar_line_boxes)
        eprint("Found " + str(len(bar_line_boxes)) + " bar lines")

        debug.write_bounding_boxes(
            "anchor_input", symbols.staff_fragments + bar_line_boxes + symbols.clefs_keys
        )
        staffs = detect_staff(
            debug,
            predictions.staff,
            symbols.staff_fragments,
            symbols.clefs_keys,
            bar_line_boxes,
            predictions.original,
        )
        if len(staffs) == 0:
            raise Exception("No staffs found")
        debug.write_bounding_boxes_alternating_colors("staffs", staffs)

        global_unit_size = np.mean([staff.average_unit_size for staff in staffs])

        bar_lines_found = add_bar_lines_to_staffs(staffs, bar_line_boxes)
        eprint("Found " + str(len(bar_lines_found)) + " bar lines")

        possible_rests = [
            rest for rest in bar_lines_or_rests if not rest.is_overlapping_with_any(bar_line_boxes)
        ]
        rests = add_rests_to_staffs(staffs, possible_rests)
        eprint("Found", len(rests), "rests")

        all_classified = predictions.notehead + predictions.clefs_keys + predictions.stems_rest
        brace_dot_img = prepare_brace_dot_image(
            predictions.symbols, predictions.staff, all_classified, global_unit_size
        )
        debug.write_threshold_image("brace_dot", brace_dot_img)
        brace_dot = create_rotated_bounding_boxes(
            brace_dot_img, skip_merging=True, max_size=(100, 1000)
        )

        notes = add_notes_to_staffs(
            staffs, noteheads_with_stems, predictions.symbols, predictions.notehead
        )
        accidentals = add_accidentals_to_staffs(staffs, symbols.accidentals)
        eprint("Found", len(accidentals), "accidentals")

        multi_staffs = find_braces_brackets_and_grand_staff_lines(debug, staffs, brace_dot)
        eprint(
            "Found",
            len(multi_staffs),
            "connected staffs (after merging grand staffs, multiple voices): ",
            [len(staff.staffs) for staff in multi_staffs],
        )

        debug.write_all_bounding_boxes_alternating_colors(
            "notes", multi_staffs, notes, rests, accidentals
        )

        title = detect_title(debug, staffs[0])
        eprint("Found title: " + title)

        result_staffs = parse_staffs(debug, multi_staffs, predictions)

        result_staffs = maintain_accidentals(result_staffs)

        eprint("Writing XML")
        xml = generate_xml(result_staffs, title)
        xml.write(xml_file)

        eprint("Finished parsing " + str(len(staffs)) + " staffs")
        teaser_file = replace_extension(image_path, "_teaser.png")
        debug.write_teaser(teaser_file, staffs)
        debug.clean_debug_files_from_previous_runs()

        eprint("Result was written to", xml_file)

        return xml_file, title, teaser_file
    except:
        if os.path.exists(xml_file):
            os.remove(xml_file)
        raise
    finally:
        debug.clean_debug_files_from_previous_runs()


def main_model2(predictions, debug, imagePath='bach1001_2.png', fdebug=False, fcache=False) -> None:
    process_image(imagePath, fdebug, fcache, predictions, debug)


if __name__ == "__main__":
    main_model2()
