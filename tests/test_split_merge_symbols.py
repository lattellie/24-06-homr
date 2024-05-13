import unittest

from training.transformer.split_merge_symbols import (
    convert_alter_to_accidentals,
    merge_symbols,
    split_symbols,
)

predlift = [
    [
        "nonote",
        "nonote",
        "nonote",
        "lift_N",
        "nonote",
        "lift_N",
        "nonote",
        "lift_N",
        "nonote",
        "lift_null",
        "nonote",
        "lift_null",
        "lift_N",
        "lift_N",
        "nonote",
        "lift_N",
        "nonote",
        "lift_N",
        "nonote",
        "lift_#",
    ]
]
predpitch = [
    [
        "nonote",
        "nonote",
        "nonote",
        "note-C4",
        "nonote",
        "note-F4",
        "nonote",
        "note-G4",
        "nonote",
        "note-B4",
        "nonote",
        "note-B4",
        "note-C5",
        "note-D5",
        "nonote",
        "note-C5",
        "nonote",
        "note-G4",
        "nonote",
        "note-E4",
    ]
]
predryhthm = [
    [
        "clef-G2",
        "keySignature-EM",
        "timeSignature-6/8",
        "note-half.",
        "barline",
        "note-half.",
        "barline",
        "note-half.",
        "barline",
        "note-half.",
        "barline",
        "note-half",
        "note-eighth",
        "note-eighth",
        "barline",
        "note-eighth",
        "|",
        "note-eighth",
        "|",
        "note-eighth",
    ]
]
prednotes = [
    [
        "nonote",
        "nonote",
        "nonote",
        "note",
        "nonote",
        "note",
        "nonote",
        "note",
        "nonote",
        "note",
        "nonote",
        "note",
        "note",
        "note",
        "nonote",
        "note",
        "nonote",
        "note",
        "nonote",
        "note",
    ]
]
merged = [
    "clef-G2+keySignature-EM+timeSignature-6/8+note-C4_half.+barline+note-F4_half.+barline+note-G4_half.+barline+note-B4_half.+barline+note-B4_half+note-C5_eighth+note-D5_eighth+barline+note-E4#_eighth|note-G4_eighth|note-C5_eighth"
]


class TestMergeSymbols(unittest.TestCase):

    def test_merge(self) -> None:
        actual = merge_symbols(predryhthm, predpitch, predlift)
        expected = convert_alter_to_accidentals(merged)
        self.assertEqual(actual, expected)

    def test_split(self) -> None:
        # Replace the + with \t as this is what the input provides
        actuallift, actualpitch, actualrhythm, actualnotes = split_symbols(
            [merged[0].replace("+", "\t")]
        )
        self.assertEqual(actualrhythm, predryhthm)
        self.assertEqual(actuallift, predlift)
        self.assertEqual(actualpitch, predpitch)
        self.assertEqual(actualnotes, prednotes)

    def test_split_sorts_notes(self) -> None:
        # Replace the + with \t as this is what the input provides
        _actuallift, actualpitch, _actualrhythm, _actualnotes = split_symbols(
            [
                "note-E4#_eighth|note-G4_eighth|note-C5_eighth\tnote-C5_eighth|note-E4#_eighth|note-G4_eighth"
            ]
        )
        self.assertEqual(
            actualpitch,
            [
                [
                    "note-C5",
                    "nonote",
                    "note-G4",
                    "nonote",
                    "note-E4",
                    "note-C5",
                    "nonote",
                    "note-G4",
                    "nonote",
                    "note-E4",
                ]
            ],
        )

    def test_split_restores_accidentals(self) -> None:
        """
        The semantic encoding doesn't tell us which accidentals are present in the image.
        The best we can do is to restore this information from
        the lift symbols and the key information.
        """

        merged_accidentals = [
            "clef-G2 keySignature-FM timeSignature-4/4 rest-sixteenth note-A3_sixteenth note-C4_sixteenth note-F4_sixteenth note-A4_sixteenth note-C4_sixteenth note-F4_sixteenth rest-sixteenth note-A3_sixteenth note-A3_sixteenth note-C4_sixteenth note-F4_sixteenth note-A4_sixteenth note-C4_sixteenth note-F4_sixteenth rest-sixteenth note-A3_sixteenth rest-sixteenth note-A3_quarter.. note-A3_quarter.. barline rest-sixteenth note-C4_sixteenth note-Eb4_sixteenth note-F4_sixteenth note-C5_sixteenth note-Eb4_sixteenth note-F4_sixteenth rest-sixteenth note-C4_sixteenth note-C4_sixteenth note-D4_sixteenth note-F#4_sixteenth note-C5_sixteenth note-D4_sixteenth note-F#4_sixteenth rest-sixteenth note-C4_sixteenth rest-sixteenth note-C4_quarter.. note-C4_quarter.. barline rest-sixteenth note-C4_sixteenth note-D4_sixteenth note-A4_sixteenth note-C5_sixteenth note-D4_sixteenth note-A4_sixteenth rest-sixteenth note-C4_sixteenth note-Bb3_sixteenth note-D4_sixteenth note-G4_sixteenth note-Bb4_sixteenth note-D4_sixteenth note-G4_sixteenth rest-sixteenth note-Bb3_sixteenth rest-sixteenth note-C4_quarter.. note-Bb3_quarter.. "  # noqa: E501
        ]
        actuallift, actualpitch, _actualrhythm, _actualnotes = split_symbols(merged_accidentals)
        readable_lift = [
            actualpitch[0][i] + lift
            for i, lift in enumerate(actuallift[0])
            if lift not in ("nonote", "lift_null")
        ]
        self.assertEqual(readable_lift, ["note-E4lift_b", "note-F4lift_#"])

    def test_split_restores_natural(self) -> None:
        """
        Bugfix: Natural symbols were not persent in the training set.
        """

        merged_accidentals = [
            "clef-G2 keySignature-GM timeSignature-4/4 note-C4_sixteenth note-F4_sixteenth note-F4_sixteenth"  # noqa: E501
        ]
        actuallift, actualpitch, _actualrhythm, _actualnotes = split_symbols(merged_accidentals)
        readable_lift = [
            actualpitch[0][i] + lift for i, lift in enumerate(actuallift[0]) if lift != "nonote"
        ]
        self.assertEqual(readable_lift, ["note-C4lift_null", "note-F4lift_N", "note-F4lift_null"])

    def test_replace_multirests(self) -> None:
        merged_multirests = [
            "multirest-1 multirest-2 multirest-3 multirest-50 multirest-100 rest-whole2"
        ]
        _actuallift, _actualpitch, actualrhythm, _actualnotes = split_symbols(merged_multirests)
        self.assertEqual(
            actualrhythm,
            [
                [
                    "rest-whole",
                    "multirest-2",
                    "multirest-3",
                    "multirest-50",
                    "multirest-50",
                    "multirest-2",
                ]
            ],
        )

    def test_accidentals_dont_affect_octaves(self) -> None:
        merged_accidentals = ["clef-G2 keySignature-CM note-F#4_quarter note-F#3_quarter"]
        actuallift, actualpitch, _actualrhythm, _actualnotes = split_symbols(merged_accidentals)
        readable_lift = [
            actualpitch[0][i] + lift for i, lift in enumerate(actuallift[0]) if lift != "nonote"
        ]
        self.assertEqual(readable_lift, ["note-F4lift_#", "note-F3lift_#"])
