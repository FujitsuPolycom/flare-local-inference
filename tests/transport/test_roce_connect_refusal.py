"""A relay group that disagrees on its wire geometry is refused before it connects.
The driver compiles the production proxy against a fake verbs provider, so it needs
no RDMA hardware and no GPU. It covers ring presence, `ring_protocol`, the ABI 12
geometry, the lane count and the two-SGE device; `roce.rs` refuses a mixed ABI first.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "helios/native/roce_proxy.c"


@pytest.fixture(scope="module")
def refusal_driver(tmp_path_factory):
    compiler = shutil.which("cc") or pytest.skip("the driver needs a C compiler")
    driver = tmp_path_factory.mktemp("roce-refusal") / "refusal"
    subprocess.run(
        [compiler, "-std=gnu11", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread",
         f'-DROCE_PROXY_SOURCE="{SOURCE}"',
         str(Path(__file__).with_name("_roce_connect_refusal.c")), "-o", str(driver)],
        check=True, capture_output=True, text=True,
    )
    return driver


def test_a_disagreeing_group_never_connects(refusal_driver):
    result = subprocess.run([str(refusal_driver)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "roce connect refusal: ok"
