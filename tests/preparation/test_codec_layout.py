# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

from google.protobuf.json_format import ParseDict
from ignition.preparation import contract, runtime_core_pb2 as wire


class CodecLayoutTest(unittest.TestCase):
    def test_a_recipe_without_physical_views_stays_raw(self):
        pool = SimpleNamespace(layout=SimpleNamespace(row_bytes=32))
        self.assertIsNone(contract.codec_layout(pool, [], {}))

    def test_only_a_recipe_declaration_selects_the_packet_profile(self):
        row = SimpleNamespace(name="layer0.recurrent", shape=(1, 128, 128), dtype="torch.float32",
                              offset=256, nbytes=65536)
        pool = SimpleNamespace(layout=SimpleNamespace(regions=[row]))
        name = "cache.0.main_k_cache"

        class Page:
            shape = (4, 64, 2, 256)
            dtype = "torch.float8_e4m3fn"

            def __getitem__(self, index):
                return SimpleNamespace(is_contiguous=lambda: True, numel=lambda: 32768)

            def element_size(self):
                return 1

        regions = [dict(storage=contract.state_view(name, 32768))]
        self.assertIsNone(contract.codec_layout(pool, regions, {name: Page()}))
        encodings = {row.name: 1, name: 42}
        encoded = contract.codec_layout(pool, regions, {name: Page()},
                                        field_encoding=lambda field, dtype, shape: encodings[field])
        layout = ParseDict(encoded, wire.CodecLayout())
        self.assertEqual(layout.revision, 1)
        self.assertEqual([(f.buffer, f.offset, f.length, f.encoding) for f in layout.fields],
                         [("state_pool", 256, 65536, 1), (name, 0, 32768, 42)])
