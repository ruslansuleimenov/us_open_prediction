import pytest

HEADER = (
    "Tournament,Date,Series,Court,Surface,Round,Best of,"
    "Player_1,Player_2,Winner,Rank_1,Rank_2,Pts_1,Pts_2,Odd_1,Odd_2,Score"
)


@pytest.fixture
def write_csv(tmp_path):
    def _write(*rows: str):
        path = tmp_path / "matches.csv"
        path.write_text("\n".join([HEADER, *rows]) + "\n")
        return path
    return _write

