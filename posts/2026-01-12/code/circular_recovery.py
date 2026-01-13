"""Circular recovery pattern for LLM-recoverable errors.

Usage:
    from circular_recovery import build_circular_workflow

    app = build_circular_workflow()
    result = app.invoke({"email_content": "...", "sender_email": "user@example.com"})
"""

from typing import Literal

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from email_agent import EmailAgentState, read_email, send_reply
from error_simulation import should_simulate_crm_error

# Initialize LLM for agent decisions
llm = ChatOllama(model="llama3.2")


class AgentDecision(BaseModel):
    """Agent's decision on next action based on current state.

    This schema structures the LLM's decision-making process when examining
    the workflow state. The LLM must provide reasoning and choose the next
    appropriate node to execute.
    """

    reasoning: str = Field(..., description="Explanation of why this action was chosen")
    next_action: Literal[
        "get_customer_history", "normalize_email", "draft_response"
    ] = Field(..., description="The next node to execute")


def agent(
    state: EmailAgentState,
) -> Command[Literal["get_customer_history", "normalize_email", "draft_response"]]:
    """LLM-powered central agent that sees errors and decides recovery strategy.

    Examines the current state, detects if there are errors, and uses the LLM to decide
    what action to take next. The agent handles three scenarios:
    1. Error detected → choose recovery action (normalize_email)
    2. No data yet → fetch data (get_customer_history)
    3. Valid data available → proceed (draft_response)

    Args:
        state: Current EmailAgentState with customer_history that may contain errors

    Returns:
        Command routing to the chosen next node
    """

    # Determine current status for clearer prompt
    has_error = bool(state.customer_history and state.customer_history.get("error"))
    has_valid_data = bool(
        state.customer_history
        and state.customer_history.get("tier")
        and not state.customer_history.get("error")
    )
    is_empty = not bool(state.customer_history)

    # Build a very explicit prompt based on current state
    if has_error:
        # Clear instruction when there's an error
        prompt = f"""
        The customer lookup failed with this error: {state.customer_history.get("error")}

        The email address is: {state.sender_email}

        You must choose: normalize_email

        This will convert the email to lowercase and retry the lookup.
        """

    elif is_empty:
        # Clear instruction when we need to fetch data
        prompt = f"""
        We need to fetch customer history for: {state.sender_email}

        Customer history is currently empty.

        You must choose: get_customer_history
        """

    elif has_valid_data:
        # Clear instruction when we have data
        prompt = f"""We have valid customer history:
        - Tier: {state.customer_history.get("tier")}
        - Account age: {state.customer_history.get("account_age_days")} days
        - Customer ID: {state.customer_history.get("customer id", "N/A")}
        - Customer name: {state.customer_history.get("customer name", "N/A")}

        You must choose: draft_response"""

    else:
        # Fallback: shouldn't reach here
        prompt = f"""Current state is unclear. Customer history: {state.customer_history}

        Default to: get_customer_history"""

    # Use structured LLM to get decision
    structured_llm = llm.with_structured_output(AgentDecision)
    decision = structured_llm.invoke(prompt)

    return Command(goto=decision.next_action)


def get_customer_history_circular(state: EmailAgentState) -> Command[Literal["agent"]]:
    """Fetch customer history - ALWAYS routes back to agent (success or failure).

    This is the key to circular recovery: instead of failing immediately,
    we store the error in state and route back to the agent, which can
    then decide on a recovery strategy.

    Note: This demo uses hard-coded customer data. Replace with actual CRM API call.

    Args:
        state: Current EmailAgentState with sender_email

    Returns:
        Command with customer data (or error) and routing back to agent
    """
    # Simulate error on first attempt only (mixed case email fails)
    if should_simulate_crm_error():
        customer_data = {
            "error": "Customer not found - email may need normalization",
            "attempted_email": state.sender_email,
        }
    else:
        customer_data = {
            "tier": "premium",
            "account_age_days": 90,
            "total_tickets": 5,
            "customer id": "CUST-7890",
            "customer name": "Jane Doe",
        }

    # KEY: Always route back to agent so it can see the result
    return Command(
        update={"customer_history": customer_data},
        goto="agent",  # Circular: let agent decide what to do next
    )


def normalize_email(state: EmailAgentState) -> Command[Literal["agent"]]:
    """Normalize email address (lowercase) and clear error.

    This is a recovery action that the agent can choose. It fixes the email
    format and clears the error state, allowing the agent to retry the lookup.

    Args:
        state: Current EmailAgentState with sender_email

    Returns:
        Command with normalized email and cleared error, routing back to agent
    """
    original_email = state.sender_email
    normalized_email = original_email.lower()

    # Clear the error and update email
    return Command(
        update={
            "sender_email": normalized_email,
            "customer_history": {},  # Clear error so agent will retry
        },
        goto="agent",  # Route back to agent to retry lookup
    )


def draft_response_with_customer_data(
    state: EmailAgentState,
) -> Command[Literal["send_reply"]]:
    """Draft response using customer history (should have data by now).

    This node expects customer_history to contain valid data (no errors).
    It uses the LLM to draft a personalized response based on customer tier.

    Args:
        state: EmailAgentState with populated customer_history

    Returns:
        Command with draft_response and routing to send_reply
    """
    customer_tier = state.customer_history.get("tier", "standard")

    # Use LLM to draft personalized response
    draft_prompt = f"""Draft a professional response to this customer email:

    Email: {state.email_content}
    Customer tier: {customer_tier}

    Create a helpful, personalized response that acknowledges their {customer_tier} status."""

    response = llm.invoke(draft_prompt)
    draft = response.content

    return Command(update={"draft_response": draft}, goto="send_reply")


def build_circular_workflow() -> StateGraph:
    """Build workflow with circular routing for LLM-recoverable errors.

    This workflow demonstrates the circular recovery pattern where:
    1. read_email → agent (start)
    2. agent → decides based on state → get_customer_history OR normalize_email OR draft_response
    3. get_customer_history → agent (circular)
    4. normalize_email → agent (circular)
    5. draft_response → send_reply → END

    The agent node acts as a hub that all recovery paths return to.

    Returns:
        Compiled StateGraph with circular routing
    """
    workflow = StateGraph(EmailAgentState)

    # Add nodes
    workflow.add_node("read_email", read_email)
    workflow.add_node("agent", agent)  # Decision hub
    workflow.add_node("get_customer_history", get_customer_history_circular)
    workflow.add_node("normalize_email", normalize_email)
    workflow.add_node("draft_response", draft_response_with_customer_data)
    workflow.add_node("send_reply", send_reply)

    # Add edges: only define the starting path
    # All other routing happens via Command returns (circular logic)
    workflow.add_edge(START, "read_email")
    workflow.add_edge("read_email", "agent")  # Route to agent after reading email
    workflow.add_edge("send_reply", END)

    return workflow.compile()
