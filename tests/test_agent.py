"""Unit test home for agent. IMPLEMENT — CI runs these."""
def test_agent_placeholder():
    from unittest.mock import Mock, call

    from doc_agent import hooks
    from doc_agent.agent.agent import Agent
    from doc_agent.contracts import Chunk
    from doc_agent.llm.prompts import ABSTAIN

    hooks.clear()
    retrieve_cfg = {"k": 10, "k_step": 10, "k_max": 40, "weak_threshold": 0.35}
    strong_chunk = Chunk(
        id="dolil_38#c1",
        doc_id="deed_p0038",
        text="Plot 2163 belongs to Rahim Uddin.",
        page_ids=["dolil_38"],
        score=0.90,
    )

    strong_retriever = Mock()
    strong_retriever.cfg = retrieve_cfg
    strong_retriever.retrieve.return_value = [strong_chunk]
    strong_agent = Agent(
        {
            "agent": {
                "max_steps": 8,
                "backend": "fake",
                "fake_responses": ["Rahim Uddin owns plot 2163 [dolil_38#c1]"],
            }
        },
        strong_retriever,
    )
    strong_answer = strong_agent.run("Who owns plot 2163?")
    strong_retriever.retrieve.assert_called_once_with("Who owns plot 2163?", 10)
    assert strong_answer.grounded is True
    assert [citation.chunk_id for citation in strong_answer.citations] == ["dolil_38#c1"]

    recovering_retriever = Mock()
    recovering_retriever.cfg = retrieve_cfg
    recovering_retriever.retrieve.side_effect = [
        [strong_chunk.model_copy(update={"score": 0.20})],
        [strong_chunk],
    ]
    recovering_agent = Agent(
        {
            "agent": {
                "max_steps": 8,
                "backend": "fake",
                "fake_responses": ["Rahim Uddin owns plot 2163 [dolil_38#c1]"],
            }
        },
        recovering_retriever,
    )
    recovered_answer = recovering_agent.run("Who owns plot 2163?")
    assert recovering_retriever.retrieve.call_args_list == [
        call("Who owns plot 2163?", 10),
        call("Who owns plot 2163?", 20),
    ]
    assert recovered_answer.grounded is True

    weak_retriever = Mock()
    weak_retriever.cfg = retrieve_cfg
    weak_retriever.retrieve.return_value = [strong_chunk.model_copy(update={"score": 0.20})]
    weak_agent = Agent(
        {"agent": {"max_steps": 8, "backend": "fake"}},
        weak_retriever,
    )
    weak_answer = weak_agent.run("Who owns an absent plot?")
    assert weak_retriever.retrieve.call_args_list == [
        call("Who owns an absent plot?", 10),
        call("Who owns an absent plot?", 20),
        call("Who owns an absent plot?", 30),
        call("Who owns an absent plot?", 40),
    ]
    assert weak_answer.text == ABSTAIN
    assert weak_answer.grounded is False
    assert weak_answer.citations == []
