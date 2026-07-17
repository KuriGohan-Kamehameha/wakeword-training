from __future__ import annotations

import random
import unittest

import generate_dataset


class PositiveSelectionTests(unittest.TestCase):
    def test_generic_and_personalized_ceilings_are_separate_and_total_is_bounded(self) -> None:
        generic = {"piper-a": [f"generic-{index}.wav" for index in range(20)]}
        personalized = {"avaas": [f"personal-{index}.wav" for index in range(8)]}
        selected_generic, selected_personal = generate_dataset.select_positive_sets(
            generic,
            personalized,
            max_total=10,
            min_per_source=1,
            min_generic=4,
            min_personalized=3,
            max_personalized=4,
            rng=random.Random(42),
        )
        self.assertEqual(len(selected_personal), 4)
        self.assertEqual(len(selected_generic), 6)
        self.assertEqual(len(selected_generic) + len(selected_personal), 10)
        self.assertTrue(all(item.startswith("generic-") for item in selected_generic))
        self.assertTrue(all(item.startswith("personal-") for item in selected_personal))

    def test_generic_diversity_cannot_be_displaced_by_personalized_samples(self) -> None:
        generic = {"piper": ["generic.wav"]}
        personalized = {"avaas": [f"personal-{index}.wav" for index in range(10)]}
        with self.assertRaisesRegex(SystemExit, "generic"):
            generate_dataset.select_positive_sets(
                generic,
                personalized,
                max_total=8,
                min_per_source=1,
                min_generic=3,
                min_personalized=3,
                max_personalized=5,
                rng=random.Random(42),
            )

    def test_personalized_minimum_is_enforced_only_when_configured(self) -> None:
        generic = {"piper": [f"generic-{index}.wav" for index in range(8)]}
        selected_generic, selected_personal = generate_dataset.select_positive_sets(
            generic,
            {},
            max_total=6,
            min_per_source=1,
            min_generic=3,
            min_personalized=0,
            max_personalized=4,
            rng=random.Random(42),
        )
        self.assertEqual(len(selected_generic), 6)
        self.assertEqual(selected_personal, [])
        with self.assertRaisesRegex(SystemExit, "personalized"):
            generate_dataset.select_positive_sets(
                generic,
                {"avaas": ["only-one.wav"]},
                max_total=6,
                min_per_source=1,
                min_generic=3,
                min_personalized=2,
                max_personalized=4,
                rng=random.Random(42),
            )


if __name__ == "__main__":
    unittest.main()
