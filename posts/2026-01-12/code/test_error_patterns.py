#!/usr/bin/env python
"""Test runner for all 4 error handling patterns.

Usage:
    python test_error_patterns.py           # Run all tests
    python test_error_patterns.py --test 1  # Run specific test
"""

import argparse
from typing import Literal

from dotenv import load_dotenv
from langgraph.types import Command, interrupt

from circular_recovery import build_circular_workflow
from email_agent import EmailAgentState, SearchAPIError
from error_simulation import (
    configure_search_failure,
    reset_all_simulations,
    search_error_counter,
    should_simulate_email_service_error,
    should_simulate_search_error,
)
from workflow_builders import build_workflow

load_dotenv()


class EmailServiceError(Exception):
    """Exception raised when email service encounters an unexpected error."""

    pass


def test_transient_retry():
    """Test 1: Transient errors with RetryPolicy.

    Demonstrates:
    - RetryPolicy with exponential backoff
    - Search fails 2 times, succeeds on 3rd attempt
    - Alternative: All retries exhausted (can be tested by increasing max_failures)
    """
    print("\nTest 1: Transient Errors with RetryPolicy")

    # Reset simulations
    reset_all_simulations()

    # Configure to fail 2 times before succeeding
    configure_search_failure(max_failures=2)

    # Create test version of search_documentation with error simulation
    def search_documentation_test(
        state: EmailAgentState,
    ) -> Command[Literal["draft_response"]]:
        """Search with simulated transient errors."""

        # Check if should simulate error
        if should_simulate_search_error():
            attempt_num = search_error_counter["count"]
            raise SearchAPIError(f"Simulated transient error (attempt {attempt_num})")
        else:
            search_error_counter["count"] = 0  # Reset for next test

        # Normal search logic
        search_results = [
            "Reset password via Settings > Security > Change Password",
            "Password must be at least 12 characters",
        ]

        return Command(update={"search_results": search_results}, goto="draft_response")

    # Build workflow with test version
    import error_simulation as sim

    sim.SIMULATE_SEARCH_ERROR = True

    app = build_workflow(
        nodes_override={"search_documentation": search_documentation_test},
        with_checkpointer=False,
    )

    initial_state = {
        "email_content": "I forgot my password, how do I reset it?",
        "sender_email": "user@example.com",
    }

    try:
        result = app.invoke(initial_state)
        print("Test 1 passed")
    except SearchAPIError as e:
        print(f"Test 1 failed: {e}")
    finally:
        reset_all_simulations()


def test_llm_recoverable():
    """Test 2: LLM-recoverable errors with circular routing.

    Demonstrates:
    - LLM agent as decision hub
    - CRM lookup fails (case sensitivity)
    - Agent detects error and chooses normalize_email
    - Normalized email succeeds on retry
    - Circular routing (nodes route back to agent)
    """
    print("\nTest 2: LLM-Recoverable Errors with Circular Routing")

    reset_all_simulations()

    import error_simulation as sim

    sim.SIMULATE_CRM_ERROR = True

    app = build_circular_workflow()

    initial_state = {
        "email_content": "I need help with my account settings",
        "sender_email": "User@Example.com",  # Mixed case: will fail first lookup
    }

    try:
        result = app.invoke(initial_state)
        print("Test 2 passed")
    except Exception as e:
        print(f"Test 2 failed: {e}")
    finally:
        reset_all_simulations()


def test_interrupt_resume():
    """Test 3: User-fixable errors with interrupt/resume.

    Demonstrates:
    - Detect missing customer_id
    - Use interrupt() to request human input
    - Resume workflow with provided data
    - Recursive pattern (node calls itself after interrupt)
    """
    print("\nTest 3: User-Fixable Errors with Interrupt/Resume")

    reset_all_simulations()

    # Create test version of search_documentation with interrupt pattern
    def search_documentation_interrupt(
        state: EmailAgentState,
    ) -> Command[Literal["search_documentation", "draft_response"]]:
        """Search with interrupt if customer_id is missing."""

        # Check if customer_id is missing
        if not state.customer_id:
            # Interrupt to ask user for customer_id
            user_input = interrupt(
                {
                    "message": "Customer ID needed",
                    "request": "Please provide the customer's account ID to look up their history",
                    "email_id": state.email_id,
                    "sender_email": state.sender_email,
                }
            )

            # Update state with customer_id and recursively call this node again
            return Command(
                update={"customer_id": user_input["customer_id"]},
                goto="search_documentation",  # Recursive: goto self
            )

        search_results = [
            "Reset password via Settings > Security > Change Password",
            "Customer tier: premium - Premium support available",
        ]

        return Command(update={"search_results": search_results}, goto="draft_response")

    app = build_workflow(
        nodes_override={"search_documentation": search_documentation_interrupt},
        with_checkpointer=True,  # Required for interrupt/resume
    )

    config = {"configurable": {"thread_id": "test-3"}}

    initial_state = {
        "email_content": "I need help resetting my password",
        "sender_email": "user@example.com",
        # Note: No customer_id will trigger interrupt
    }

    try:
        result = app.invoke(initial_state, config=config)

        # Resume with customer_id
        result = app.invoke(
            Command(resume={"customer_id": "CUST-12345"}),
            config=config,
        )
        print("Test 3 passed")

    except Exception as e:
        print(f"Test 3 failed: {e}")
        import traceback

        traceback.print_exc()


def test_unexpected_error():
    """Test 4: Unexpected errors that bubble up.

    Demonstrates:
    - Unexpected error (email service 500 error)
    - Log state context for debugging
    - Re-raise exception (fail fast)
    - LangSmith captures full state
    """
    print("\nTest 4: Unexpected Errors (Bubble Up)")

    reset_all_simulations()

    # Create test version of send_reply with unexpected error
    def send_reply_test4(state: EmailAgentState) -> dict:
        """Send reply - unexpected errors bubble up."""
        try:
            # Check for simulated error
            if should_simulate_email_service_error(state.sender_email):
                raise EmailServiceError(
                    "Email service returned 500: Internal Server Error - "
                    "Domain blacklist service unavailable"
                )
            return {"reply_sent": True}

        except EmailServiceError:
            # Log context and re-raise
            raise

    import error_simulation as sim

    sim.SIMULATE_EMAIL_SERVICE_ERROR = True

    app = build_workflow(
        nodes_override={"send_reply": send_reply_test4},
        with_checkpointer=False,
    )

    initial_state = {
        "email_content": "I need help with my account",
        "sender_email": "user@blocked-domain.com",  # Will trigger error
    }

    try:
        _ = app.invoke(initial_state)
        print("Test 4 failed: should have raised an exception")
    except EmailServiceError:
        print("Test 4 passed")
    finally:
        reset_all_simulations()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run error handling pattern tests")
    parser.add_argument(
        "--test", type=int, choices=[1, 2, 3, 4], help="Run specific test (1-4)"
    )
    args = parser.parse_args()

    if args.test:
        # Run specific test
        tests = {
            1: test_transient_retry,
            2: test_llm_recoverable,
            3: test_interrupt_resume,
            4: test_unexpected_error,
        }
        tests[args.test]()
    else:
        # Run all tests

        test_transient_retry()
        test_llm_recoverable()
        test_interrupt_resume()
        test_unexpected_error()
        print("\nAll tests complete.")


if __name__ == "__main__":
    main()
