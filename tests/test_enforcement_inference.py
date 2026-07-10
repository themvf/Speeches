import run_financial_news_pipeline as core


def test_non_enforcement_speech_is_not_classified():
    # "in order to" contains the whole word "order"; "supercharges" contains
    # "charges" — neither should classify a general speech as enforcement.
    result = core._infer_enforcement_metadata(
        title="Remarks on Market Structure Reform",
        text="We must act in order to protect investors. This supercharges innovation.",
        source_kind="sec_speech",
    )
    assert result["action_type"] == "unknown"
    assert result["forum"] == "unknown"
    assert result["outcome_status"] == "unknown"
    assert result["alleged_violations"] == []


def test_speech_mentioning_securities_act_is_not_a_violation():
    result = core._infer_enforcement_metadata(
        title="Testimony on the Securities Act and Exchange Act",
        text="The Securities Act and Exchange Act frame our disclosure regime.",
        source_kind="federal_reserve_speech_testimony",
    )
    assert result["alleged_violations"] == []
    assert result["respondents"] == []
    assert result["entities"] == []


def test_enforcement_source_kind_classifies_even_with_thin_text():
    result = core._infer_enforcement_metadata(
        title="SEC Charges Advisor",
        text="The Commission charged the respondent.",
        source_kind="sec_enforcement_litigation",
    )
    assert result["action_type"] == "filing"


def test_enforcement_context_in_text_enables_classification():
    result = core._infer_enforcement_metadata(
        title="DOJ Announces Settlement",
        text="The defendant settled fraud charges and agreed to pay a penalty in U.S. district court.",
        source_kind="doj_usao_press_release",
    )
    assert result["action_type"] in {"settlement", "filing"}
    assert result["forum"] == "federal_court"


def test_supercharges_does_not_trigger_filing_in_enforcement_context():
    # Even within an enforcement doc, "supercharges" must not read as "charges".
    result = core._infer_enforcement_metadata(
        title="SEC settlement announced",
        text="The firm settled. Its new product supercharges returns for clients.",
        source_kind="sec_enforcement_litigation",
    )
    # "settled" -> settlement, not "filing" from supercharges.
    assert result["action_type"] == "settlement"
