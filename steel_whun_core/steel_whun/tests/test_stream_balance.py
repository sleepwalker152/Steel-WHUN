import pytest

from steel_whun import EnergyNetwork, EnergySource, TransmissionEdge


def test_source_energy_and_exergy_closure() -> None:
    network = EnergyNetwork()
    network.add_source(
        EnergySource(
            source_id="heat",
            carrier="hot_water",
            temperature=400.0,
            pressure=101325.0,
            max_energy_rate=1000.0,
            exergy_factor=0.40,
        )
    )
    network.add_edge(
        TransmissionEdge(
            source_id="heat",
            target_id="sink",
            energy_efficiency=0.90,
            exergy_efficiency=0.80,
            capacity=1000.0,
        )
    )

    energy_map, summary = network.dispatch({"heat": {"sink": 600.0}})
    stream = energy_map["sink"][0]

    assert summary["feasible"]
    assert summary["total_energy_requested"] == pytest.approx(600.0)
    assert summary["total_energy_delivered"] == pytest.approx(540.0)
    assert summary["total_exergy_delivered"] == pytest.approx(192.0)
    assert summary["total_energy_loss"] == pytest.approx(60.0)
    assert summary["total_exergy_loss"] == pytest.approx(48.0)
    assert stream.energy_rate == pytest.approx(540.0)
    assert stream.exergy_rate == pytest.approx(192.0)
    assert 600.0 == pytest.approx(stream.energy_rate + summary["total_energy_loss"])
    assert 240.0 == pytest.approx(stream.exergy_rate + summary["total_exergy_loss"])
