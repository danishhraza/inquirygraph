from inquirygraph.agent.graph import route_after_coverage


def test_route_loops_when_more_research_needed():
    state = {
        "needs_more_research": True,
        "pending_tasks": [{"id": "t1"}],
    }
    assert route_after_coverage(state) == "gather_and_index"


def test_route_synthesizes_when_done():
    state = {
        "needs_more_research": False,
        "pending_tasks": [],
    }
    assert route_after_coverage(state) == "detect_contradictions"

def test_route_loops_back_when_done_but_tasks_remain():
    state = {
        "needs_more_research": True,
        "pending_tasks": [{"id": "t1"}],
    }
    assert route_after_coverage(state) == "gather_and_index"
