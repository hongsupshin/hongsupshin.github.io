"""Workflow construction helpers for testing LangGraph agents.

Usage:
    from workflow_builders import build_workflow

    app = build_workflow()
    app_test = build_workflow(nodes_override={"search_documentation": custom_search})
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from email_agent import (
    EmailAgentState,
    bug_tracking,
    classify_intent,
    draft_response,
    human_review,
    read_email,
    search_documentation,
    send_reply,
)


def build_workflow(
    nodes_override: dict = None, with_checkpointer: bool = True
) -> StateGraph:
    """Build email agent workflow with optional node overrides.

    This helper function is essential for testing because LangGraph workflows
    are compiled (once compiled, node functions are locked in). To test different
    behaviors (like error simulations), we must rebuild the workflow with modified
    functions.

    Args:
        nodes_override: Dict of {"node_name": node_function} to override defaults.
                       Example: {"search_documentation": search_documentation_test}
        with_checkpointer: Whether to include memory checkpointer for state
                          persistence. Set to False for simple tests that don't
                          need interrupt/resume functionality.

    Returns:
        Compiled StateGraph ready for execution

    Example:
        # Build workflow with test version of search_documentation
        def search_documentation_test(state):
            # Simulate search error
            raise SearchAPIError("Simulated error")

        app_test = build_workflow(
            nodes_override={"search_documentation": search_documentation_test}
        )

        # Run test
        result = app_test.invoke({"email_content": "test email"})
    """
    # Default nodes: production implementations from email_agent
    default_nodes = {
        "read_email": read_email,
        "classify_intent": classify_intent,
        "search_documentation": search_documentation,
        "bug_tracking": bug_tracking,
        "draft_response": draft_response,
        "human_review": human_review,
        "send_reply": send_reply,
    }

    # Override with test versions if provided
    nodes = {**default_nodes, **(nodes_override or {})}

    # Create workflow
    workflow = StateGraph(EmailAgentState)

    # Add nodes with special handling for search_documentation (has RetryPolicy)
    for name, func in nodes.items():
        if name == "search_documentation":
            # Apply retry policy to search node (transient error pattern)
            workflow.add_node(
                name,
                func,
                retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0),
            )
        else:
            workflow.add_node(name, func)

    # Add edges: define the flow
    workflow.add_edge(START, "read_email")
    workflow.add_edge("read_email", "classify_intent")
    # classify_intent uses Command for conditional routing
    # search_documentation and bug_tracking use Command for routing
    workflow.add_edge("send_reply", END)

    if with_checkpointer:
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    return workflow.compile()


def build_simple_workflow_for_test(
    node_function, node_name: str = "test_node"
) -> StateGraph:
    """Build a minimal workflow for testing a single node in isolation.

    Useful for unit testing individual nodes without the full workflow complexity.

    Args:
        node_function: The node function to test
        node_name: Name for the node (default: "test_node")

    Returns:
        Compiled StateGraph with START -> node -> END

    Example:
        def test_search(state):
            return {"search_results": ["test result"]}

        app = build_simple_workflow_for_test(test_search, "search_documentation")
        result = app.invoke({"email_content": "test"})
        assert "search_results" in result
    """
    workflow = StateGraph(EmailAgentState)
    workflow.add_node(node_name, node_function)
    workflow.add_edge(START, node_name)
    workflow.add_edge(node_name, END)
    return workflow.compile()
