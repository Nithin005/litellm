"""
Unit tests for GithubCopilotAnthropicConfig and its integration with the routing layer.
"""
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# validate_environment tests
# ---------------------------------------------------------------------------

def test_validate_environment_sets_bearer_auth():
    """Should use Authorization: Bearer (not x-api-key) with the Copilot token."""
    from litellm.llms.github_copilot.chat.anthropic_transformation import (
        GithubCopilotAnthropicConfig,
    )

    config = GithubCopilotAnthropicConfig()
    fake_key = "ghu_testtoken123"

    with patch(
        "litellm.llms.github_copilot.chat.anthropic_transformation.Authenticator"
    ) as MockAuth:
        MockAuth.return_value.get_api_key.return_value = fake_key
        headers = config.validate_environment(
            headers={},
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hello"}],
            optional_params={},
            litellm_params={},
        )

    assert headers.get("Authorization") == f"Bearer {fake_key}"
    assert "x-api-key" not in headers
    assert headers.get("anthropic-version") == "2023-06-01"
    assert "editor-version" in headers
    assert "copilot-integration-id" in headers


def test_validate_environment_adds_effort_beta_for_non_4_6():
    """Non-4.6 Claude models with reasoning_effort should get the effort beta header."""
    from litellm.llms.github_copilot.chat.anthropic_transformation import (
        GithubCopilotAnthropicConfig,
    )

    config = GithubCopilotAnthropicConfig()
    fake_key = "ghu_testtoken123"

    with patch(
        "litellm.llms.github_copilot.chat.anthropic_transformation.Authenticator"
    ) as MockAuth:
        MockAuth.return_value.get_api_key.return_value = fake_key
        headers = config.validate_environment(
            headers={},
            model="claude-opus-4-5",
            messages=[{"role": "user", "content": "think"}],
            optional_params={"reasoning_effort": "high"},
            litellm_params={},
        )

    assert "effort-2025-11-24" in headers.get("anthropic-beta", "")


def test_validate_environment_no_effort_beta_for_claude_4_6():
    """Claude 4.6 uses the stable API — no effort beta header should be set."""
    from litellm.llms.github_copilot.chat.anthropic_transformation import (
        GithubCopilotAnthropicConfig,
    )

    config = GithubCopilotAnthropicConfig()
    fake_key = "ghu_testtoken123"

    with patch(
        "litellm.llms.github_copilot.chat.anthropic_transformation.Authenticator"
    ) as MockAuth:
        MockAuth.return_value.get_api_key.return_value = fake_key
        headers = config.validate_environment(
            headers={},
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "think"}],
            optional_params={"output_config": {"effort": "high"}, "reasoning_effort": "high"},
            litellm_params={},
        )

    assert "effort-2025-11-24" not in headers.get("anthropic-beta", "")


# ---------------------------------------------------------------------------
# ProviderConfigManager routing tests
# ---------------------------------------------------------------------------

def test_provider_config_manager_returns_anthropic_config_for_claude():
    """ProviderConfigManager should return GithubCopilotAnthropicConfig for Claude models."""
    import litellm
    from litellm.utils import ProviderConfigManager
    from litellm.llms.github_copilot.chat.anthropic_transformation import (
        GithubCopilotAnthropicConfig,
    )

    ProviderConfigManager._PROVIDER_CONFIG_MAP = None  # reset cache

    cfg = ProviderConfigManager.get_provider_chat_config(
        "claude-sonnet-4-5", litellm.LlmProviders.GITHUB_COPILOT
    )
    assert isinstance(cfg, GithubCopilotAnthropicConfig)


def test_provider_config_manager_returns_openai_config_for_gpt():
    """ProviderConfigManager should return GithubCopilotConfig for non-Claude models."""
    import litellm
    from litellm.utils import ProviderConfigManager
    from litellm.llms.github_copilot.chat.transformation import GithubCopilotConfig

    ProviderConfigManager._PROVIDER_CONFIG_MAP = None  # reset cache

    cfg = ProviderConfigManager.get_provider_chat_config(
        "gpt-4o", litellm.LlmProviders.GITHUB_COPILOT
    )
    assert isinstance(cfg, GithubCopilotConfig)


# ---------------------------------------------------------------------------
# main.py routing test
# ---------------------------------------------------------------------------

def test_copilot_claude_routes_through_anthropic_handler():
    """litellm.completion with github_copilot/claude-* should call the Anthropic handler."""
    import litellm

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock(content="test")
    mock_response.model = "claude-sonnet-4-5"

    with (
        patch("litellm.main.anthropic_chat_completions") as mock_anthropic,
        patch("litellm.main.openai_chat_completions") as mock_openai,
        patch(
            "litellm.llms.github_copilot.authenticator.Authenticator.get_api_key",
            return_value="ghu_fake",
        ),
        patch(
            "litellm.llms.github_copilot.authenticator.Authenticator.get_api_base",
            return_value="https://api.githubcopilot.com",
        ),
    ):
        mock_anthropic.completion.return_value = mock_response

        litellm.completion(
            model="github_copilot/claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hello"}],
        )

    mock_anthropic.completion.assert_called_once()
    mock_openai.completion.assert_not_called()

    kwargs = mock_anthropic.completion.call_args.kwargs
    assert kwargs.get("api_base", "").endswith("/v1/messages"), (
        f"Expected api_base to end with /v1/messages, got: {kwargs.get('api_base')}"
    )
    assert kwargs.get("custom_llm_provider") == "github_copilot"
