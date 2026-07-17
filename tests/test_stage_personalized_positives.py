from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import avaas_import
import stage_personalized_positives as stager
from tests.fixture_builder import build_export, sha256, write_tone


class PersonalizedStagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        bundle = build_export(self.root, clip_count=10)
        imported = avaas_import.import_bundle(
            bundle,
            self.root / "imports",
            allow_nonpromotable=True,
        )
        self.imported = imported.path
        self.train = self.root / "run/hey_piranesi/positive_train"
        self.test = self.root / "run/hey_piranesi/positive_test"
        for directory, offset in ((self.train, 0), (self.test, 10)):
            directory.mkdir(parents=True)
            for index in range(3):
                write_tone(
                    directory / f"generic-piper-{offset + index}.wav",
                    frequency=300 + offset + index,
                )
        self.generic_hashes = {
            path: sha256(path)
            for path in (*self.train.glob("generic-*.wav"), *self.test.glob("generic-*.wav"))
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_stages_deterministic_split_into_actual_upstream_directories(self) -> None:
        first = stager.stage_exports(
            [self.imported],
            positive_train=self.train,
            positive_test=self.test,
            seed=42,
            min_generic=6,
            min_personalized=8,
            max_personalized=64,
            max_total=100,
            allow_nonpromotable=True,
        )
        self.assertEqual(first.personalized_count, 10)
        self.assertEqual(first.generic_count, 6)
        self.assertGreater(first.train_count, 0)
        self.assertGreater(first.test_count, 0)
        for path, digest in self.generic_hashes.items():
            self.assertEqual(sha256(path), digest)
        staged = sorted((*self.train.glob("avaas_*.wav"), *self.test.glob("avaas_*.wav")))
        self.assertEqual(len(staged), 10)
        self.assertTrue((self.train.parent / "avaas-personalized-stage.json").is_file())

        second = stager.stage_exports(
            [self.imported],
            positive_train=self.train,
            positive_test=self.test,
            seed=42,
            min_generic=6,
            min_personalized=8,
            max_personalized=64,
            max_total=100,
            allow_nonpromotable=True,
        )
        self.assertTrue(second.idempotent)
        self.assertEqual(first.mapping_sha256, second.mapping_sha256)
        self.assertEqual(len((*list(self.train.glob("avaas_*.wav")), *list(self.test.glob("avaas_*.wav")))), 10)

    def test_split_is_reproducible_and_seed_bound(self) -> None:
        first = stager.plan_split([self.imported], seed=42, allow_nonpromotable=True)
        again = stager.plan_split([self.imported], seed=42, allow_nonpromotable=True)
        changed = stager.plan_split([self.imported], seed=99, allow_nonpromotable=True)
        self.assertEqual(first, again)
        self.assertNotEqual(first, changed)
        self.assertEqual({item["split"] for item in first}, {"train", "test"})

    def test_refuses_to_replace_generic_diversity_or_exceed_bounds(self) -> None:
        with self.assertRaisesRegex(stager.StagingError, "generic"):
            stager.stage_exports(
                [self.imported],
                positive_train=self.train,
                positive_test=self.test,
                min_generic=7,
                min_personalized=8,
                max_personalized=64,
                max_total=100,
                allow_nonpromotable=True,
            )
        with self.assertRaisesRegex(stager.StagingError, "personalized|maximum"):
            stager.stage_exports(
                [self.imported],
                positive_train=self.train,
                positive_test=self.test,
                min_generic=6,
                min_personalized=8,
                max_personalized=9,
                max_total=100,
                allow_nonpromotable=True,
            )
        with self.assertRaisesRegex(stager.StagingError, "total"):
            stager.stage_exports(
                [self.imported],
                positive_train=self.train,
                positive_test=self.test,
                min_generic=6,
                min_personalized=8,
                max_personalized=64,
                max_total=15,
                allow_nonpromotable=True,
            )

    def test_duplicate_personalized_audio_across_exports_is_rejected(self) -> None:
        second = self.root / "imports/second-export-v1"
        import shutil

        shutil.copytree(self.imported, second)
        manifest = second / "manifest.json"
        raw = manifest.read_text(encoding="utf-8").replace(
            '"export_id":"hey-piranesi-fixture-v1"', '"export_id":"second-export-v1"'
        )
        manifest.write_text(raw, encoding="utf-8")
        from tests.fixture_builder import seal

        seal(second)
        with self.assertRaisesRegex(stager.StagingError, "duplicate"):
            stager.plan_split(
                [self.imported, second], seed=42, allow_nonpromotable=True
            )

    def test_trainer_invokes_stager_after_generate_before_normalize(self) -> None:
        trainer = (Path(__file__).resolve().parents[1] / "trainer.sh").read_text(
            encoding="utf-8"
        )
        generate = trainer.index("    run_generate_clips\n")
        stage = trainer.index('python3 "$avaas_stager"')
        normalize = trainer.index("    CFG_OUT=\"$cfg_out\" python3 - <<'PY'", generate)
        self.assertLess(generate, stage)
        self.assertLess(stage, normalize)


if __name__ == "__main__":
    unittest.main()
