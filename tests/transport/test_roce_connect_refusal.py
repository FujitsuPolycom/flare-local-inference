"""A RoCE group that disagrees on its wire record is refused before it connects.

The driver compiles the production proxy against a fake verbs provider, so it needs
no RDMA hardware and no GPU. It covers ring presence, the ABI 14 lane count in the
record, and the create-time lane bound, at the one lane this runtime opens and at a
wider group. `roce.rs` refuses a core recording a proxy ABI this runtime does not
load before any of this runs.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "helios/native/roce_proxy.c"


@pytest.fixture(scope="module")
def compiler():
    cc = shutil.which("cc") or pytest.skip("the driver needs a C compiler")
    probe = subprocess.run(
        [cc, "-E", "-x", "c", "-"],
        input="#include <infiniband/verbs.h>\n",
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        pytest.skip("the driver needs the libibverbs development headers")
    return cc


@pytest.fixture(scope="module")
def refusal_driver(compiler, tmp_path_factory):
    directory = tmp_path_factory.mktemp("roce-refusal")

    def build(lanes):
        driver = directory / f"refusal-{lanes}"
        subprocess.run(
            [compiler, "-std=gnu11", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread",
             f"-DLANES={lanes}",
             f'-DROCE_PROXY_SOURCE="{SOURCE}"',
             str(Path(__file__).with_name("_roce_connect_refusal.c")), "-o", str(driver)],
            check=True, capture_output=True, text=True,
        )
        return driver

    return build


@pytest.mark.parametrize("lanes", [1, 4])
def test_a_disagreeing_group_never_connects(refusal_driver, lanes):
    result = subprocess.run([str(refusal_driver(lanes))], capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "roce connect refusal: ok"
