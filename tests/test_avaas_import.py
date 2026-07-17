from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from tests.fixture_builder import build_export, read_json, seal, write_json

import avaas_import


class AvaasImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bundle = build_export(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_validates_exact_phrase_identity_audio_and_provenance(self) -> None:
        result = avaas_import.validate_bundle(self.bundle)
        self.assertEqual(result.export_id, "hey-piranesi-fixture-v1")
        self.assertEqual(result.phrase, "Hey Piranesi")
        self.assertEqual(result.alias, "piranesi")
        self.assertEqual(result.presentation_id, "piranesi.en-gb.neutral")
        self.assertEqual(result.clip_count, 10)
        self.assertFalse(result.promotable)

    def test_rejects_unknown_schema_traversal_symlink_and_extra_file(self) -> None:
        manifest = read_json(self.bundle / "manifest.json")
        manifest["schema"] = "avaas/wakeword-export@v999"
        write_json(self.bundle / "manifest.json", manifest)
        seal(self.bundle)
        with self.assertRaisesRegex(avaas_import.ImportContractError, "schema"):
            avaas_import.validate_bundle(self.bundle)

        self.bundle = build_export(self.root, export_id="checksum-escape-v1")
        lines = (self.bundle / "checksums.sha256").read_text(encoding="ascii").splitlines()
        digest = lines[0].split("  ", 1)[0]
        lines[0] = f"{digest}  ../outside"
        (self.bundle / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")
        os.utime(self.bundle / "READY", None)
        with self.assertRaisesRegex(avaas_import.ImportContractError, "path|inventory|checksum"):
            avaas_import.validate_bundle(self.bundle)

        self.bundle = build_export(self.root, export_id="symlink-export-v1")
        outside = self.root / "outside.wav"
        clip = next((self.bundle / "clips").glob("*.wav"))
        outside.write_bytes(clip.read_bytes())
        clip.unlink()
        clip.symlink_to(outside)
        os.utime(self.bundle / "READY", None)
        with self.assertRaisesRegex(avaas_import.ImportContractError, "symlink"):
            avaas_import.validate_bundle(self.bundle)

        self.bundle = build_export(self.root, export_id="extra-export-v1")
        (self.bundle / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        seal(self.bundle)
        with self.assertRaisesRegex(avaas_import.ImportContractError, "unexpected|executable"):
            avaas_import.validate_bundle(self.bundle)

    def test_rejects_wrong_phrase_alias_rate_hash_and_duplicate_clip(self) -> None:
        manifest = read_json(self.bundle / "manifest.json")
        manifest["phrase"]["text"] = "Okay Piranesi"
        write_json(self.bundle / "manifest.json", manifest)
        seal(self.bundle)
        with self.assertRaisesRegex(avaas_import.ImportContractError, "phrase"):
            avaas_import.validate_bundle(self.bundle)

        self.bundle = build_export(self.root, export_id="alias-export-v1")
        manifest = read_json(self.bundle / "manifest.json")
        manifest["source_artifact"]["alias"] = "satraj"
        write_json(self.bundle / "manifest.json", manifest)
        seal(self.bundle)
        with self.assertRaisesRegex(avaas_import.ImportContractError, "alias|Piranesi"):
            avaas_import.validate_bundle(self.bundle)

        self.bundle = build_export(self.root, export_id="rate-export-v1")
        manifest = read_json(self.bundle / "manifest.json")
        manifest["audio"]["sample_rate_hz"] = 24_000
        write_json(self.bundle / "manifest.json", manifest)
        seal(self.bundle)
        with self.assertRaisesRegex(avaas_import.ImportContractError, "audio|rate"):
            avaas_import.validate_bundle(self.bundle)

        self.bundle = build_export(self.root, export_id="hash-export-v1")
        clip = next((self.bundle / "clips").glob("*.wav"))
        clip.write_bytes(clip.read_bytes() + b"tamper")
        os.utime(self.bundle / "READY", None)
        with self.assertRaisesRegex(avaas_import.ImportContractError, "checksum|audio"):
            avaas_import.validate_bundle(self.bundle)

        self.bundle = build_export(self.root, export_id="duplicate-export-v1")
        clips = read_json(self.bundle / "clips/manifest.json")
        clips["clips"].append(dict(clips["clips"][0]))
        write_json(self.bundle / "clips/manifest.json", clips)
        manifest = read_json(self.bundle / "manifest.json")
        manifest["clips"]["count"] += 1
        manifest["clips"]["total_frames"] += clips["clips"][0]["frames"]
        import hashlib

        manifest["clips"]["manifest_sha256"] = hashlib.sha256(
            (self.bundle / "clips/manifest.json").read_bytes()
        ).hexdigest()
        write_json(self.bundle / "manifest.json", manifest)
        seal(self.bundle)
        with self.assertRaisesRegex(avaas_import.ImportContractError, "duplicate"):
            avaas_import.validate_bundle(self.bundle)

    def test_import_is_append_only_idempotent_and_refuses_fixture_by_default(self) -> None:
        destination = self.root / "imports"
        with self.assertRaisesRegex(avaas_import.ImportContractError, "promotable|fixture"):
            avaas_import.import_bundle(self.bundle, destination)
        first = avaas_import.import_bundle(
            self.bundle, destination, allow_nonpromotable=True
        )
        second = avaas_import.import_bundle(
            self.bundle, destination, allow_nonpromotable=True
        )
        self.assertFalse(first.idempotent)
        self.assertTrue(second.idempotent)
        self.assertEqual(first.manifest_sha256, second.manifest_sha256)
        imported = destination / self.bundle.name
        self.assertEqual(
            avaas_import.validate_bundle(imported).manifest_sha256,
            first.manifest_sha256,
        )
        (imported / "manifest.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(avaas_import.ImportContractError, "collision|invalid|checksum"):
            avaas_import.import_bundle(
                self.bundle, destination, allow_nonpromotable=True
            )


if __name__ == "__main__":
    unittest.main()
