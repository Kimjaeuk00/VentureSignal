from core.state import evaluation_update, make_analysis, merge_evaluations


def test_merge_keeps_fields_from_other_nodes():
    tech = evaluation_update("C01", technology=make_analysis("t"))["evaluations"]
    market = evaluation_update("C01", market=make_analysis("m"))["evaluations"]

    merged = merge_evaluations(merge_evaluations({}, tech), market)

    assert merged["C01"]["technology"]["summary"] == "t"
    assert merged["C01"]["market"]["summary"] == "m"


def test_merge_adds_new_company():
    merged = merge_evaluations({"C01": {"decision": "HOLD"}}, {"C02": {"decision": "INVEST"}})
    assert set(merged) == {"C01", "C02"}
