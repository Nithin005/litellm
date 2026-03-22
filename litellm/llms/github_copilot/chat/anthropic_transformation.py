"""
GithubCopilotAnthropicConfig: routes Claude model calls on GitHub Copilot through
the Anthropic messages API (/v1/messages) with Copilot auth headers.

All Anthropic message-format transformation logic (thinking, tool calling, etc.)
is inherited from AnthropicConfig.  Only validate_environment is overridden to
swap the auth mechanism: instead of requiring ANTHROPIC_API_KEY it uses the
short-lived Copilot bearer token and adds the Copilot-specific request headers.
"""

from typing import Dict, List, Optional

from litellm.llms.github_copilot.authenticator import Authenticator
from litellm.llms.github_copilot.common_utils import GetAPIKeyError, get_copilot_default_headers
from litellm.types.llms.openai import AllMessageValues


def _get_anthropic_config_base():
    """Deferred import to avoid circular dependency at module load time."""
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig
    return AnthropicConfig


class GithubCopilotAnthropicConfig(_get_anthropic_config_base()):
    """
    AnthropicConfig subclass for GitHub Copilot.

    Overrides only validate_environment to use Copilot bearer auth instead of
    the Anthropic API key.  All request transformation (thinking params, tool
    calling format, streaming, etc.) is inherited unchanged from AnthropicConfig.
    """

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict:
        """
        Build request headers for Copilot's Anthropic messages endpoint.

        Steps:
        1. Obtain the short-lived Copilot bearer token via Authenticator.
        2. Use get_anthropic_headers(auth_token=...) to generate the standard
           Anthropic headers (anthropic-version, beta flags for thinking/effort,
           etc.) with Bearer auth instead of x-api-key.
        3. Merge with Copilot-specific headers (editor-version, copilot-integration-id,
           etc.); Copilot headers win on any key collision so that Authorization:
           Bearer (uppercase A) from get_copilot_default_headers overrides the
           lowercase authorization set by get_anthropic_headers.
        """
        try:
            copilot_api_key: str = Authenticator().get_api_key()
        except GetAPIKeyError as e:
            import litellm as _litellm
            raise _litellm.AuthenticationError(
                message=str(e),
                llm_provider="github_copilot",
                model=model,
            ) from e

        tools = optional_params.get("tools")

        # Determine which Anthropic beta headers are needed for this request.
        anthropic_headers = self.get_anthropic_headers(
            api_key=None,           # Do NOT set x-api-key
            auth_token=copilot_api_key,  # Sets authorization: Bearer <token>
            computer_tool_used=self.is_computer_tool_used(tools=tools),
            prompt_caching_set=self.is_cache_control_set(messages=messages),
            pdf_used=self.is_pdf_used(messages=messages),
            file_id_used=self.is_file_id_used(messages=messages),
            mcp_server_used=self.is_mcp_server_used(
                mcp_servers=optional_params.get("mcp_servers")
            ),
            web_search_tool_used=self.is_web_search_tool_used(tools=tools),
            tool_search_used=self.is_tool_search_used(tools=tools),
            programmatic_tool_calling_used=self.is_programmatic_tool_calling_used(
                tools=tools
            ),
            input_examples_used=self.is_input_examples_used(tools=tools),
            effort_used=self.is_effort_used(
                optional_params=optional_params, model=model
            ),
            code_execution_tool_used=self.is_code_execution_tool_used(tools=tools),
            container_with_skills_used=self.is_container_with_skills_used(
                optional_params=optional_params
            ),
            user_anthropic_beta_headers=self._get_user_anthropic_beta_headers(
                anthropic_beta_header=headers.get("anthropic-beta")
            ),
        )

        # Copilot-specific headers (Authorization: Bearer, editor-version, etc.)
        # Merged last so they take precedence over any overlapping Anthropic headers.
        copilot_headers = get_copilot_default_headers(copilot_api_key)

        return {**headers, **anthropic_headers, **copilot_headers}
