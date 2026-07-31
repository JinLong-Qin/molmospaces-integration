import pytest

from molmo_spaces.data_generation.run_validation import require_success_count


@pytest.mark.parametrize(
    ("success_count", "total_count", "required_count"),
    [(0, 0, 1), (1, 4, 2)],
)
def test_require_success_count_rejects_shortfall(
    success_count,
    total_count,
    required_count,
):
    with pytest.raises(RuntimeError, match=rf"produced {success_count}.*required at least"):
        require_success_count(success_count, total_count, required_count)


@pytest.mark.parametrize("required_count", [None, 1, 2])
def test_require_success_count_accepts_met_or_disabled_gate(required_count):
    require_success_count(2, 2, required_count)


def test_require_success_count_rejects_invalid_requirement():
    with pytest.raises(ValueError, match="at least 1"):
        require_success_count(0, 0, 0)
