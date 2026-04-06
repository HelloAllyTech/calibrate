"""
Unit tests for agent connection benchmarking.

Covers:
- model/provider params included in request body
- model/provider threaded through run_test_external
- verify() sends model/provider for benchmark verify
- OpenRouter model name parsing
- output folder naming per model

Run with:
    python -m pytest tests/test_agent_benchmarking.py -v
"""

import unittest
from unittest.mock import patch, AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Reuse the mock helper from test_agent_connection
# ---------------------------------------------------------------------------

def _make_httpx_response(body: dict, status: int = 200):
    import httpx
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = body
    mock.raise_for_status = MagicMock()
    if status >= 400:
        mock.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status}", request=MagicMock(), response=mock
        )
    return mock


def _patch_httpx(response_body: dict, status: int = 200):
    mock_resp = _make_httpx_response(response_body, status)
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)
    return patch("httpx.AsyncClient", return_value=mock_client), mock_client


# ---------------------------------------------------------------------------
# Tests for call_text_agent — model/provider in request body
# ---------------------------------------------------------------------------

class TestCallTextAgentModelParams(unittest.IsolatedAsyncioTestCase):

    async def test_model_and_provider_included_in_body(self):
        from calibrate.connections import TextAgentConnection
        from calibrate.llm.run_tests import call_text_agent

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx({"response": "ok"})
        with ctx:
            await call_text_agent(
                [{"role": "user", "content": "Hi"}],
                agent,
                model="gemma-4-26b-a4b-it",
                provider="google",
            )

        body = mock_client.post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "gemma-4-26b-a4b-it")
        self.assertEqual(body["provider"], "google")
        self.assertIn("messages", body)

    async def test_model_and_provider_absent_when_not_passed(self):
        from calibrate.connections import TextAgentConnection
        from calibrate.llm.run_tests import call_text_agent

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx({"response": "ok"})
        with ctx:
            await call_text_agent(
                [{"role": "user", "content": "Hi"}],
                agent,
            )

        body = mock_client.post.call_args.kwargs["json"]
        self.assertNotIn("model", body)
        self.assertNotIn("provider", body)
        self.assertIn("messages", body)

    async def test_only_model_included_when_provider_omitted(self):
        from calibrate.connections import TextAgentConnection
        from calibrate.llm.run_tests import call_text_agent

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx({"response": "ok"})
        with ctx:
            await call_text_agent(
                [{"role": "user", "content": "Hi"}],
                agent,
                model="gpt-4o",
            )

        body = mock_client.post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "gpt-4o")
        self.assertNotIn("provider", body)


# ---------------------------------------------------------------------------
# Tests for run_test_external — model/provider threaded through
# ---------------------------------------------------------------------------

class TestRunTestExternalModelParams(unittest.IsolatedAsyncioTestCase):

    async def test_model_and_provider_passed_to_call_text_agent(self):
        from calibrate.connections import TextAgentConnection
        from calibrate.llm.run_tests import run_test_external

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx({"response": "Sure, the weather is sunny."})

        evaluation = {"type": "response", "criteria": "Agent answers the question"}
        mock_judge = AsyncMock(return_value={"match": True, "reasoning": "ok"})

        with ctx, patch("calibrate.llm.run_tests.test_response_llm_judge", mock_judge):
            await run_test_external(
                chat_history=[{"role": "user", "content": "What's the weather?"}],
                evaluation=evaluation,
                agent=agent,
                model="gemma-4-26b-a4b-it",
                provider="google",
            )

        body = mock_client.post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "gemma-4-26b-a4b-it")
        self.assertEqual(body["provider"], "google")

    async def test_no_model_params_when_not_passed(self):
        from calibrate.connections import TextAgentConnection
        from calibrate.llm.run_tests import run_test_external

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx({"response": "hello"})
        mock_judge = AsyncMock(return_value={"match": True, "reasoning": "ok"})

        with ctx, patch("calibrate.llm.run_tests.test_response_llm_judge", mock_judge):
            await run_test_external(
                chat_history=[{"role": "user", "content": "Hi"}],
                evaluation={"type": "response", "criteria": "greet"},
                agent=agent,
            )

        body = mock_client.post.call_args.kwargs["json"]
        self.assertNotIn("model", body)
        self.assertNotIn("provider", body)


# ---------------------------------------------------------------------------
# Tests for TextAgentConnection.verify() — benchmark verify
# ---------------------------------------------------------------------------

class TestVerifyWithModelParams(unittest.IsolatedAsyncioTestCase):

    async def test_verify_includes_model_and_provider(self):
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx({"response": "hi"})
        with ctx:
            result = await agent.verify(model="gemma-4-26b-a4b-it", provider="google")

        self.assertTrue(result["ok"])
        body = mock_client.post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "gemma-4-26b-a4b-it")
        self.assertEqual(body["provider"], "google")

    async def test_verify_without_model_params_has_only_messages(self):
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx({"response": "hi"})
        with ctx:
            result = await agent.verify()

        self.assertTrue(result["ok"])
        body = mock_client.post.call_args.kwargs["json"]
        self.assertNotIn("model", body)
        self.assertNotIn("provider", body)
        self.assertIn("messages", body)

    async def test_verify_passes_even_when_agent_ignores_model_params(self):
        """Agent that returns valid format regardless of model params should pass."""
        from calibrate.connections import TextAgentConnection

        agent = TextAgentConnection(url="http://fake-agent/chat")
        ctx, mock_client = _patch_httpx({"response": "I am using gemma", "tool_calls": []})
        with ctx:
            result = await agent.verify(model="gemma-4-26b-a4b-it", provider="google")

        self.assertTrue(result["ok"])


# ---------------------------------------------------------------------------
# Tests for _parse_openrouter_model (CLI helper)
# ---------------------------------------------------------------------------

class TestParseOpenrouterModel(unittest.TestCase):

    def _parse(self, model_str, provider_arg="openrouter"):
        from calibrate.cli import _parse_openrouter_model
        return _parse_openrouter_model(model_str, provider_arg)

    def test_openrouter_format_splits_correctly(self):
        self.assertEqual(self._parse("google/gemma-4-26b-a4b-it"), ("google", "gemma-4-26b-a4b-it"))

    def test_openai_slash_format(self):
        self.assertEqual(self._parse("openai/gpt-4o"), ("openai", "gpt-4o"))

    def test_plain_model_uses_provider_arg(self):
        self.assertEqual(self._parse("gpt-4o", "openai"), ("openai", "gpt-4o"))

    def test_plain_model_defaults_to_openai_when_no_provider(self):
        self.assertEqual(self._parse("gpt-4o", ""), ("openai", "gpt-4o"))

    def test_nested_slash_only_splits_on_first(self):
        # anthropic/claude-3-5-sonnet-20241022 → provider=anthropic, model=claude-3-5-sonnet-20241022
        self.assertEqual(
            self._parse("anthropic/claude-3-5-sonnet-20241022"),
            ("anthropic", "claude-3-5-sonnet-20241022"),
        )


# ---------------------------------------------------------------------------
# Tests for _run_single_model folder naming
# ---------------------------------------------------------------------------

class TestFolderNaming(unittest.IsolatedAsyncioTestCase):

    async def _get_folder(self, model: str, agent=True):
        """Run _run_single_model and capture the output dir it creates."""
        import os
        import tempfile
        from calibrate.connections import TextAgentConnection
        from calibrate.llm import _Tests

        fake_agent = TextAgentConnection(url="http://fake-agent/chat") if agent else None
        fake_body = {"response": "hello", "tool_calls": []}
        test_cases = [
            {
                "history": [{"role": "user", "content": "hi"}],
                "evaluation": {"type": "response", "criteria": "greet"},
            }
        ]
        mock_judge = AsyncMock(return_value={"match": True, "reasoning": "ok"})

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx, _ = _patch_httpx(fake_body)
            with ctx, patch("calibrate.llm.run_tests.test_response_llm_judge", mock_judge):
                await _Tests._run_single_model(
                    system_prompt="",
                    tools=[],
                    test_cases=test_cases,
                    output_dir=tmpdir,
                    model=model,
                    provider="openrouter",
                    agent=fake_agent,
                )
            created = [
                d for d in os.listdir(tmpdir)
                if os.path.isdir(os.path.join(tmpdir, d))
            ]
            return created[0] if created else None

    async def test_openrouter_model_folder_uses_double_underscore(self):
        folder = await self._get_folder("google/gemma-4-26b-a4b-it")
        self.assertEqual(folder, "google__gemma-4-26b-a4b-it")

    async def test_plain_model_folder_is_model_name(self):
        folder = await self._get_folder("gpt-4o")
        self.assertEqual(folder, "gpt-4o")

    async def test_no_model_falls_back_to_external_agent(self):
        folder = await self._get_folder("")
        self.assertEqual(folder, "external_agent")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
