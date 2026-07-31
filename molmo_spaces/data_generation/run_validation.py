def require_success_count(
    success_count: int,
    total_count: int,
    required_count: int | None,
) -> None:
    """Fail a bounded datagen run that did not produce enough successes."""
    if required_count is None:
        return
    if required_count < 1:
        raise ValueError("required success count must be at least 1")
    if success_count < required_count:
        raise RuntimeError(
            f"Datagen produced {success_count} successful trajectories out of {total_count}; "
            f"required at least {required_count}"
        )
