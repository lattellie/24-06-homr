from homr import constants
from homr.results import (
    ClefType,
    ResultClef,
    ResultDuration,
    ResultMeasure,
    ResultNote,
    ResultNoteGroup,
    ResultPitch,
    ResultRest,
    ResultStaff,
    ResultTimeSignature,
)
from homr.simple_logging import eprint


class TrOMRParser:
    def __init__(self) -> None:
        self._key_signatures: list[int] = []
        self._time_signatures: list[str] = []
        self._clefs: list[ClefType] = []
        self._number_of_accidentals: int = 0

    def number_of_clefs(self) -> int:
        return len(self._clefs)

    def number_of_key_signatures(self) -> int:
        return len(self._key_signatures)

    def number_of_time_signatures(self) -> int:
        return len(self._time_signatures)

    def number_of_accidentals(self) -> int:
        """
        Returns the number of accidentals including the key signatures.
        """
        return self._number_of_accidentals + sum(
            [abs(key_signature) for key_signature in self._key_signatures]
        )

    def parse_clef(self, clef: str) -> ResultClef:
        parts = clef.split("-")
        clef_type_str = parts[1]
        clef_type = ClefType.TREBLE if clef_type_str.startswith("G") else ClefType.BASS
        self._clefs.append(clef_type)
        return ResultClef(clef_type, 0)

    def parse_key_signature(self, key_signature: str, clef: ResultClef) -> None:
        key_signature_mapping = {
            "CbM": -7,
            "GbM": -6,
            "DbM": -5,
            "AbM": -4,
            "EbM": -3,
            "BbM": -2,
            "FM": -1,
            "CM": 0,
            "GM": 1,
            "DM": 2,
            "AM": 3,
            "EM": 4,
            "BM": 5,
            "F#M": 6,
            "C#M": 7,
        }
        signature_name = key_signature.split("-")[1]
        if signature_name in key_signature_mapping:
            clef.circle_of_fifth = key_signature_mapping[signature_name]
            self._key_signatures.append(clef.circle_of_fifth)
        else:
            eprint("WARNING: Unrecognized key signature: " + signature_name)

    def parse_time_signature(self, clef: str) -> ResultTimeSignature:
        parts = clef.split("-")
        clef_type = parts[1]
        self._time_signatures.append(clef_type)
        return ResultTimeSignature(clef_type)

    def parse_duration_name(self, duration_name: str) -> int:
        duration_mapping = {
            "whole": constants.duration_of_quarter * 4,
            "half": constants.duration_of_quarter * 2,
            "quarter": constants.duration_of_quarter,
            "eighth": constants.duration_of_quarter // 2,
            "sixteenth": constants.duration_of_quarter // 4,
            "thirty_second": constants.duration_of_quarter // 8,
        }
        return duration_mapping.get(duration_name, constants.duration_of_quarter // 16)

    def _adjust_duration_with_dot(self, duration: int, has_dot: bool) -> int:
        if has_dot:
            return duration * 3 // 2
        else:
            return duration

    def parse_note(self, note: str) -> ResultNote:
        try:
            note_details = note.split("-")[1]
            pitch_and_duration = note_details.split("_")
            pitch = pitch_and_duration[0]
            duration = pitch_and_duration[1]
            has_dot = duration.endswith(".")
            duration_name = duration[:-1] if has_dot else duration
            note_name = pitch[0]
            octave = int(pitch[1])
            alter = None
            len_with_accidental = 2
            if len(pitch) > len_with_accidental:
                accidental = pitch[2]
                self._number_of_accidentals += 1
                if accidental == "b":
                    alter = -1
                elif accidental == "#":
                    alter = 1
                else:
                    alter = 0

            return ResultNote(
                ResultPitch(note_name, octave, alter),
                ResultDuration(
                    self._adjust_duration_with_dot(
                        self.parse_duration_name(duration_name), has_dot
                    ),
                    has_dot,
                ),
            )
        except Exception:
            eprint("Failed to parse note: " + note)
            return ResultNote(
                ResultPitch("C", 4, 0), ResultDuration(constants.duration_of_quarter, False)
            )

    def parse_notes(self, notes: str) -> ResultNote | ResultNoteGroup:
        note_parts = notes.split("|")
        note_parts = [note_part for note_part in note_parts if note_part.startswith("note")]
        if len(note_parts) == 1:
            return self.parse_note(note_parts[0])
        else:
            return ResultNoteGroup([self.parse_note(note_part) for note_part in note_parts])

    def parse_rest(self, rest: str) -> ResultRest:
        rest = rest.split("|")[0]
        duration = rest.split("-")[1]
        has_dot = duration.endswith(".")
        duration_name = duration[:-1] if has_dot else duration
        return ResultRest(
            ResultDuration(
                self._adjust_duration_with_dot(self.parse_duration_name(duration_name), has_dot),
                has_dot,
            )
        )

    def parse_tr_omr_output(self, output: str) -> ResultStaff:
        parts = output.split("+")
        measures = []
        current_measure = ResultMeasure([])

        parse_functions = {
            "clef": self.parse_clef,
            "timeSignature": self.parse_time_signature,
            "note": self.parse_notes,
            "rest": self.parse_rest,
        }

        for part in parts:
            if part == "barline":
                measures.append(current_measure)
                current_measure = ResultMeasure([])
            elif part.startswith("keySignature"):
                if len(current_measure.symbols) > 0 and isinstance(
                    current_measure.symbols[-1], ResultClef
                ):
                    self.parse_key_signature(part, current_measure.symbols[-1])
            elif part.startswith("multirest"):
                eprint("Skipping over multirest")
            else:
                for prefix, parse_function in parse_functions.items():
                    if part.startswith(prefix):
                        current_measure.symbols.append(parse_function(part))
                        break

        if len(current_measure.symbols) > 0:
            measures.append(current_measure)
        return ResultStaff(measures)
