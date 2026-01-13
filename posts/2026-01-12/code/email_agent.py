"""Email agent workflow for processing customer emails.

Usage:
    from email_agent import create_email_agent

    app = create_email_agent()
    result = app.invoke({"email_content": "...", "sender_email": "user@example.com"})
"""

import os
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt
from pydantic import BaseModel, Field, field_validator

load_dotenv()

# Initialize LLM
llm = ChatOllama(model="llama3.2")


class SearchAPIError(Exception):
    """Exception raised when search API encounters an error."""

    pass


class EmailClassification(BaseModel):
    """Classification results for an email."""

    intent: Literal["question", "bug", "billing", "feature", "complex"] = Field(
        ..., description="The primary intent of the email"
    )
    urgency: Literal["low", "medium", "high", "critical"] = Field(
        ..., description="Urgency level of the email"
    )
    topic: str = Field(..., description="Main topic or subject matter of the email")
    summary: str = Field(..., description="Brief summary of the email content")

    @field_validator("topic", "summary")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Ensure topic and summary are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class EmailAgentState(BaseModel):
    """State schema for email processing agent with Pydantic validation."""

    # Raw email data (required)
    email_content: str = Field(..., description="Full content of the email")
    sender_email: str = Field(..., description="Email address of the sender")
    email_id: str = Field(
        default="email-1", description="Unique identifier for the email"
    )

    # Classification result (optional, populated by classifier node)
    classification: EmailClassification | None = Field(
        default=None, description="Classification results after analyzing the email"
    )

    # Raw search/API results (optional, populated by retrieval nodes)
    search_results: list[str] = Field(
        default_factory=list, description="List of raw document chunks from search"
    )
    customer_history: dict = Field(
        default_factory=dict, description="Raw customer data from CRM system"
    )

    # Generated content (optional, populated by generation nodes)
    draft_response: str = Field(
        default="", description="Draft response generated for the email"
    )
    messages: list[Any] = Field(
        default_factory=list,
        description="Conversation or processing messages (can include HumanMessage, AIMessage, etc.)",
    )

    @field_validator("email_content")
    @classmethod
    def validate_email_content(cls, v: str) -> str:
        """Ensure email content is not empty."""
        if not v or not v.strip():
            raise ValueError("Email content cannot be empty")
        return v.strip()

    @field_validator("sender_email")
    @classmethod
    def validate_sender_email(cls, v: str) -> str:
        """Basic email format validation."""
        if not v or "@" not in v:
            raise ValueError("Invalid email address format")
        return v.strip().lower()


def read_email(state: EmailAgentState) -> dict:
    """
    Read and validate incoming email.

    Node functions accept Pydantic state models and return dicts.
    """
    return {
        "messages": [HumanMessage(content=f"Processing email: {state.email_content}")]
    }


def classify_intent(
    state: EmailAgentState,
) -> Command[
    Literal["search_documentation", "human_review", "draft_response", "bug_tracking"]
]:
    """Use LLM to classify email intent and urgency, then route accordingly"""

    # Create structured LLM that returns EmailClassification Pydantic model
    structured_llm = llm.with_structured_output(EmailClassification)

    # Format the prompt on-demand, not stored in state
    classification_prompt = f"""
    Analyze this customer email and classify it:

    Email: {state.email_content}
    From: {state.sender_email}

    Provide classification including intent, urgency, topic, and summary.
    """

    # Get structured response as Pydantic model
    classification = structured_llm.invoke(classification_prompt)

    # Determine next node based on classification (use attribute access)
    if classification.intent == "billing" or classification.urgency == "critical":
        goto = "human_review"
    elif classification.intent in ["question", "feature"]:
        goto = "search_documentation"
    elif classification.intent == "bug":
        goto = "bug_tracking"
    else:
        goto = "draft_response"

    # Store classification as a dict in state (convert Pydantic model to dict)
    return Command(update={"classification": classification.model_dump()}, goto=goto)


def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Search knowledge base for relevant information."""

    # Build search query from classification
    if state.classification:
        query = f"{state.classification.intent} {state.classification.topic}"
    else:
        query = "general query"

    print(f"Searching documentation for: {query}")

    try:
        # Implement your search logic here (e.g., using a search API, RAG, etc.)
        # Store raw search results, not formatted text
        search_results = [
            "Reset password via Settings > Security > Change Password",
            "Password must be at least 12 characters",
            "Include uppercase, lowercase, numbers, and symbols",
        ]  # multiple results from top-k search results
    except SearchAPIError as e:
        # For recoverable search errors, store error and continue
        search_results = [f"Search temporarily unavailable: {e!s}"]

    return Command(
        update={"search_results": search_results},  # Store raw results or error
        goto="draft_response",
    )


def bug_tracking(_: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Create or update bug tracking ticket.

    Note: This is a demo with hard-coded ticket ID. Replace with actual API call.
    """

    # Create ticket in your bug tracking system
    ticket_id = "BUG-12345"  # Example. Would be created via API

    return Command(
        update={"search_results": [f"Bug ticket {ticket_id} created"]},
        goto="draft_response",
    )


def draft_response(
    state: EmailAgentState,
) -> Command[Literal["human_review", "send_reply"]]:
    """Generate response using context and route based on quality"""

    # Format context from raw state data on-demand
    context_sections = []

    if state.search_results:
        # Format search results for the prompt
        formatted_docs = "\n".join([f"- {doc}" for doc in state.search_results])
        context_sections.append(f"Relevant documentation:\n{formatted_docs}")

    if state.customer_history:
        # Format customer data for the prompt
        context_sections.append(
            f"Customer tier: {state.customer_history.get('tier', 'standard')}"
        )

    # Build the prompt with formatted context (chr(10) is newline)
    if state.classification:
        intent = state.classification.intent
        urgency = state.classification.urgency
    else:
        intent = "unknown"
        urgency = "medium"

    draft_prompt = f"""
    Draft a response to this customer email:
    {state.email_content}

    Email intent: {intent}
    Urgency level: {urgency}

    {chr(10).join(context_sections)}

    Guidelines:
    - Be professional and helpful
    - Address their specific concern
    - Use the provided documentation when relevant
    """  # chr(10) is newline

    response = llm.invoke(draft_prompt)

    # Determine if human review needed based on urgency and intent
    needs_review = urgency in ["high", "critical"] or intent == "complex"

    # Route to appropriate next node
    goto = "human_review" if needs_review else "send_reply"

    return Command(
        update={"draft_response": response.content},  # Store only the raw response
        goto=goto,
    )


def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
    """Pause for human review using interrupt and route based on decision"""

    # interrupt() must come first: any code before it will re-run on resume
    if state.classification:
        urgency = state.classification.urgency
        intent = state.classification.intent
    else:
        urgency = "unknown"
        intent = "unknown"

    human_decision = interrupt(
        {
            "email_id": state.email_id,
            "original_email": state.email_content,
            "draft_response": state.draft_response,
            "urgency": urgency,
            "intent": intent,
            "action": "Please review and approve/edit this response",
        }
    )

    # Resume to process the human's decision
    if human_decision.get("approved"):
        return Command(
            update={
                "draft_response": human_decision.get(
                    "edited_response", state.draft_response
                )
            },
            goto="send_reply",
        )
    else:
        # Rejection means human will handle directly
        return Command(update={}, goto=END)


def send_reply(state: EmailAgentState) -> dict:
    """Send the email response.

    Note: This is a demo. Replace print with actual email service integration.
    """
    # Integrate with email service (e.g., SMTP, SendGrid, etc.)
    print(f"Sending reply: {state.draft_response[:100]}...")  # Print first 100 chars
    return {}


def create_email_agent():
    """Create and compile the email agent workflow"""

    # Create the graph
    workflow = StateGraph(EmailAgentState)

    # Add nodes with appropriate error handling
    workflow.add_node("read_email", read_email)
    workflow.add_node("classify_intent", classify_intent)

    # Add retry policy for nodes that might have transient failures
    workflow.add_node(
        "search_documentation",
        search_documentation,
        retry_policy=RetryPolicy(max_attempts=3),
    )
    workflow.add_node("bug_tracking", bug_tracking)
    workflow.add_node("draft_response", draft_response)
    workflow.add_node("human_review", human_review)
    workflow.add_node("send_reply", send_reply)

    # Add only the essential edges
    workflow.add_edge(START, "read_email")
    workflow.add_edge("read_email", "classify_intent")
    workflow.add_edge("send_reply", END)

    graph = workflow.compile()

    return graph


# Create the default graph instance
graph = create_email_agent()

if __name__ == "__main__":
    print("Email agent workflow created.")
    print(f"LangSmith tracing: {os.getenv('LANGCHAIN_TRACING_V2', 'false')}")
