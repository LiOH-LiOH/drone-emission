"""Portable, colour-independent data and strict replay regression checks."""

import json
import hashlib
import ast
import re
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from drone_emission.visualization.palette import configure_palette, color, opacity, DEFAULT_PALETTE
from drone_emission.visualization.prepared_bundle import (
    Codec, SourceOnly, capture, prepared_data, prepared_plot, replay,
    PLOT_FUNCTIONS,
    SCHEMA_VERSION, drawing_code_hash, file_hash, validate_bundle, write_document,
)


def palette_with_override(key, value):
    text, count = re.subn(r'^"' + re.escape(key) + r'" = .*$',
                         f'"{key}" = "{value}"', DEFAULT_PALETTE.read_text(),
                         flags=re.MULTILINE)
    assert count == 1, f"Expected one palette definition for {key}"
    return text


class PreparedBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.roots = {key: self.root / key for key in ("data", "work", "cache", "figures", "project", "package")}
        self.codec = Codec(self.root / "bundle", self.roots)
        configure_palette()

    def tearDown(self):
        configure_palette()
        self.temp.cleanup()

    def test_tables_arrays_nan_and_index_roundtrip(self):
        frame = pd.DataFrame({"city": ["上海", "北京"], "value": [1.2345678901234567, np.nan]}, index=pd.Index([4, 8], name="row"))
        encoded = self.codec.encode(frame)
        pd.testing.assert_frame_equal(frame, self.codec.decode(encoded))
        np.testing.assert_array_equal(np.array([[1, 2]], dtype=np.int16), self.codec.decode(self.codec.encode(np.array([[1, 2]], dtype=np.int16))))

    def test_geometry_and_colour_roundtrip(self):
        import geopandas as gpd
        from shapely.geometry import Polygon
        frame = gpd.GeoDataFrame({"height": [3.0]}, geometry=[Polygon([(0, 0), (1, 0), (1, 1)])], crs="EPSG:4326")
        restored = self.codec.decode(self.codec.encode(frame))
        self.assertTrue(restored.geometry.equals(frame.geometry))
        self.assertEqual(restored.crs, frame.crs)
        token = self.codec.encode(color("mode.drone"))
        self.assertEqual(token, {"type": "colour", "key": "mode.drone"})
        alternate = self.root / "palette.toml"
        alternate.write_text(palette_with_override("mode.drone", "#123456"))
        configure_palette(alternate)
        self.assertEqual(self.codec.decode(token), "#123456")
        self.assertEqual(self.codec.encode(color("mode.drone")), token)

    def test_paths_relocate_and_traversal_fails(self):
        encoded = self.codec.encode(str(self.roots["figures"] / "paper" / "plot.pdf"))
        elsewhere = Codec(self.codec.root, dict(self.roots, figures=self.root / "new-figures"))
        self.assertEqual(elsewhere.decode(encoded), str(self.root / "new-figures/paper/plot.pdf"))
        with self.assertRaises(ValueError):
            self.codec.decode({"type": "root_path", "root": "figures", "relative": "../escape"})

    def test_opacity_remains_editable_after_preparation(self):
        key = "roadnet.opacity_0_65"
        encoded = self.codec.encode(opacity(key))
        original = configure_palette()
        usage = {"colors": [], "colormaps": [], "alpha": [key]}
        alternate = self.root / "palette.toml"
        alternate.write_text(DEFAULT_PALETTE.read_text().replace(
            f'"{key}" = 0.65', f'"{key}" = 0.25'))
        changed = configure_palette(alternate)
        self.assertEqual(self.codec.decode(encoded), 0.25)
        self.assertNotEqual(original.signature_for(usage), changed.signature_for(usage))

    def test_replay_does_not_run_preparation_and_detects_call_mismatch(self):
        outputs = []
        source_calls = []

        @prepared_data
        def compute(source):
            source_calls.append(True)
            return np.array([2.0, 4.0])

        def draw(source):
            outputs.append(compute(source))

        draw.__module__ = "drone_emission.visualization.prepared_bundle"
        draw.__name__ = "_test_draw"
        draw = prepared_plot(omit=("source",))(draw)
        with capture(self.codec.root, self.roots) as session:
            draw(object())
        self.assertEqual(len(source_calls), 1)
        replay(self.codec.root, {"entries": session.entries}, self.roots)
        self.assertEqual(len(source_calls), 1)
        np.testing.assert_array_equal(outputs[0], outputs[1])
        session.entries[0]["data"][0]["function"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "call order mismatch"):
            replay(self.codec.root, {"entries": session.entries}, self.roots)
        PLOT_FUNCTIONS.pop(f"{draw.__module__}:{draw.__name__}")

    def test_unserializable_objects_fail(self):
        with self.assertRaises(TypeError):
            self.codec.encode(object())
        with self.assertRaisesRegex(RuntimeError, "raw source"):
            self.codec.decode(self.codec.encode(SourceOnly("graph"))).get_local_edges({})

    def test_bundle_corruption_and_cohort_are_rejected(self):
        root = self.root / "prepared"
        request = {"id": "sample", "cache_version": 1, "outputs": ["sample.pdf"]}
        directory = root / "sample"
        directory.mkdir(parents=True)
        document = {
            "schema_version": SCHEMA_VERSION, "request_id": "sample",
            "request_version": 1, "outputs": request["outputs"],
            "cohort_sha256": "cohort-a", "drawing_code_sha256": drawing_code_hash(),
            "files": {}, "entries": [],
        }
        document["payload_sha256"] = hashlib.sha256(json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()
        write_document(directory / "prepared.json", document)
        validate_bundle(root, request, "cohort-a")
        with self.assertRaisesRegex(ValueError, "cohort_sha256"):
            validate_bundle(root, request, "cohort-b")
        document["entries"] = ["corrupted"]
        write_document(directory / "prepared.json", document)
        with self.assertRaisesRegex(ValueError, "Corrupt prepared document"):
            validate_bundle(root, request, "cohort-a")

    def test_raw_read_is_fatal_even_if_plot_catches_it(self):
        self.roots["data"].mkdir()
        source = self.roots["data"] / "forbidden.csv"
        source.write_text("do not read")

        def draw(source):
            try:
                Path(source).read_text()
            except RuntimeError:
                pass

        draw.__module__ = "drone_emission.visualization.prepared_bundle"
        draw.__name__ = "_test_forbidden_draw"
        draw = prepared_plot()(draw)
        entry = {"function": f"{draw.__module__}:{draw.__name__}",
                 "arguments": self.codec.encode({"source": str(source)}), "data": []}
        with self.assertRaisesRegex(RuntimeError, "forbidden input access"):
            replay(self.codec.root, {"entries": [entry]}, self.roots)
        PLOT_FUNCTIONS.pop(entry["function"])

    def test_palette_extraction_does_not_rename_artist_methods(self):
        root = Path(__file__).parents[1] / "src/drone_emission/visualization"
        for source in root.glob("*.py"):
            for node in ast.walk(ast.parse(source.read_text())):
                if isinstance(node, ast.Attribute):
                    self.assertNotIn("paper_color", node.attr, (source.name, node.lineno))

    def test_city_binned_cache_ignores_display_order_but_not_source_changes(self):
        from drone_emission.visualization.city_gef_detour import _canonical_binned_cache_meta
        first = {"city_keys": ["beijing", "chengdu"],
                 "merchant_filter": {"region_keys": ["beijing", "chengdu"], "quantile": 0.99},
                 "source_files": {"beijing": {"mtime_ns": 123}}}
        second = {"city_keys": ["chengdu", "beijing"],
                  "merchant_filter": {"region_keys": ["chengdu", "beijing"], "quantile": 0.99},
                  "source_files": {"beijing": {"mtime_ns": 123}}}
        self.assertEqual(_canonical_binned_cache_meta(first), _canonical_binned_cache_meta(second))
        second["source_files"]["beijing"]["mtime_ns"] = 124
        self.assertNotEqual(_canonical_binned_cache_meta(first), _canonical_binned_cache_meta(second))
        self.assertEqual(second["city_keys"], ["chengdu", "beijing"])

    def test_palette_signature_tracks_alias_and_colormap_dependencies(self):
        original = configure_palette()
        usage = {"colors": ["roadnet.1f77b4"], "colormaps": []}
        signature = original.signature_for(usage)
        alternate = self.root / "palette.toml"
        alternate.write_text(palette_with_override("mode.drone", "#123456"))
        self.assertNotEqual(configure_palette(alternate).signature_for(usage), signature)
        alternate.write_text(palette_with_override("mode.fuel_car", "#123456"))
        self.assertEqual(configure_palette(alternate).signature_for(usage), signature)


if __name__ == "__main__":
    unittest.main()
