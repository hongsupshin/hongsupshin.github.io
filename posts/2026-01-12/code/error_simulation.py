"""Error simulation infrastructure for testing LangGraph agents.

Usage:
    from error_simulation import configure_search_failure, reset_all_simulations

    configure_search_failure(max_failures=2)
    # Run your test...
    reset_all_simulations()
"""

# Flag for transient search errors (Test 1)
SIMULATE_SEARCH_ERROR = False

# Flag for CRM lookup errors (Test 2: LLM-recoverable)
SIMULATE_CRM_ERROR = False

# Flag for email service errors (Test 3: interrupt pattern)
SIMULATE_EMAIL_SERVICE_ERROR = False

# Counter for search API errors (transient error pattern)
search_error_counter = {
    "count": 0,  # Current attempt number
    "max_failures": 2,  # How many times to fail before succeeding
}

# Counter for CRM lookup attempts (LLM-recoverable pattern)
crm_lookup_counter = {
    "count": 0,  # Current attempt number
}


def configure_search_failure(max_failures: int = 2) -> None:
    """Configure how many times search should fail before succeeding.

    Args:
        max_failures: Number of times to fail (default: 2 means fail twice,
                     succeed on 3rd attempt)
    """
    search_error_counter["max_failures"] = max_failures
    search_error_counter["count"] = 0


def configure_crm_failure(fail_on_attempt: int = 1) -> None:
    """Configure which CRM lookup attempt should fail.

    Args:
        fail_on_attempt: Attempt number to fail on (default: 1 means first attempt fails)
    """
    # For now, CRM always fails on attempt 1 when SIMULATE_CRM_ERROR is True
    # This can be extended if needed for more complex test scenarios
    crm_lookup_counter["count"] = 0


def configure_email_service_failure(enabled: bool = True) -> None:
    """Configure email service simulation.

    Args:
        enabled: Whether to simulate email service errors
    """
    global SIMULATE_EMAIL_SERVICE_ERROR
    SIMULATE_EMAIL_SERVICE_ERROR = enabled


def reset_all_simulations() -> None:
    """Reset all simulation flags and counters to initial state.

    This should be called between tests to ensure clean state.
    """
    global SIMULATE_SEARCH_ERROR, SIMULATE_CRM_ERROR, SIMULATE_EMAIL_SERVICE_ERROR

    # Reset flags
    SIMULATE_SEARCH_ERROR = False
    SIMULATE_CRM_ERROR = False
    SIMULATE_EMAIL_SERVICE_ERROR = False

    # Reset counters
    search_error_counter["count"] = 0
    search_error_counter["max_failures"] = 2
    crm_lookup_counter["count"] = 0


def should_simulate_search_error() -> bool:
    """Check if a search error should be simulated on this attempt.

    Returns:
        True if error should be raised, False if operation should succeed
    """
    if not SIMULATE_SEARCH_ERROR:
        return False

    search_error_counter["count"] += 1
    return search_error_counter["count"] <= search_error_counter["max_failures"]


def should_simulate_crm_error() -> bool:
    """Check if a CRM error should be simulated on this attempt.

    Returns:
        True if error should be raised, False if operation should succeed
    """
    if not SIMULATE_CRM_ERROR:
        return False

    crm_lookup_counter["count"] += 1
    # Fail on first attempt only
    return crm_lookup_counter["count"] == 1


def should_simulate_email_service_error(email: str) -> bool:
    """Check if an email service error should be simulated.

    Args:
        email: Email address being sent to

    Returns:
        True if error should be raised, False if operation should succeed
    """
    if not SIMULATE_EMAIL_SERVICE_ERROR:
        return False

    # Simulate error for blocked domains
    return email.endswith("@blocked-domain.com")
