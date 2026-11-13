"""Transport setups the proxy must refuse before it can post a byte.
The driver compiles the production proxy against a fake verbs provider, so it needs
no RDMA hardware and no GPU. It covers the four disagreements a relay group can
carry at connect and the two-SGE device requirement, then admits an all-gather relay
plan only where forwards are direct: a mesh and a closed four-rank ring accept one, a
line and rings of three and five refuse it. `roce.rs` refuses a mixed ABI first.
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


@pytest.mark.parametrize("mode, expected",
                         [([], "roce connect refusal: ok"), (["relay"], "roce relay admission: ok")])
def test_the_proxy_refuses_what_it_cannot_carry(refusal_driver, mode, expected):
    result = subprocess.run([str(refusal_driver), *mode], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == expected
