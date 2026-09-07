"""Portable, colour-independent data and strict replay regression checks."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from drone_emission.visualization.palette import configure_palette, color, DEFAULT_PALETTE
from drone_emission.visualization.prepared_bundle import (
    Codec, SourceOnly, capture, prepared_data, prepared_plot, replay,
    PLOT_FUNCTIONS,
)


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
        alternate.write_text(DEFAULT_PALETTE.read_text().replace('"mode.drone" = "#0072B2"', '"mode.drone" = "#123456"'))
        configure_palette(alternate)
        self.assertEqual(self.codec.decode(token), "#123456")
        self.assertEqual(self.codec.encode(color("mode.drone")), token)

    def test_paths_relocate_and_traversal_fails(self):
        encoded = self.codec.encode(str(self.roots["figures"] / "paper" / "plot.pdf"))
        elsewhere = Codec(self.codec.root, dict(self.roots, figures=self.root / "new-figures"))
        self.assertEqual(elsewhere.decode(encoded), str(self.root / "new-figures/paper/plot.pdf"))
        with self.assertRaises(ValueError):
            self.codec.decode({"type": "root_path", "root": "figures", "relative": "../escape"})

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


if __name__ == "__main__":
    unittest.main()
