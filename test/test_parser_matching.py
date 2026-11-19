# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

from kv_capacity_estimator.tokenize.parser_matching import (
    detect_tool_parser_mismatch,
    log_tool_parser_mismatch,
)


def config(architectures=(), model_type=None, text_model_type=None):
    text_config = (
        SimpleNamespace(model_type=text_model_type) if text_model_type else None
    )
    return SimpleNamespace(
        architectures=list(architectures),
        model_type=model_type,
        text_config=text_config,
    )


class ToolParserMatchingTest(unittest.TestCase):
    def test_glm_config_with_deepseek_parser_is_flagged(self):
        message = detect_tool_parser_mismatch(
            "test_model/GLM-5.3/",
            "deepseekv41",
            hf_config=config(
                architectures=["GlmMoeDsaForCausalLM"], model_type="glm_moe_dsa"
            ),
        )
        self.assertIsNotNone(message)
        self.assertIn("'deepseekv41'", message)
        self.assertIn("glm", message)
        self.assertIn("deepseek", message)
        self.assertIn("glm47", message)

    def test_glm_model_with_glm_parser_matches(self):
        self.assertIsNone(
            detect_tool_parser_mismatch(
                "test_model/GLM-5.3/",
                "glm47",
                hf_config=config(
                    architectures=["GlmMoeDsaForCausalLM"], model_type="glm_moe_dsa"
                ),
            )
        )
        self.assertIsNone(
            detect_tool_parser_mismatch("models/GLM-5.2-FP8", "glm")
        )

    def test_deepseek_config_with_deepseek_parser_matches(self):
        self.assertIsNone(
            detect_tool_parser_mismatch(
                "models/DeepSeek-V4.1-Flash",
                "deepseekv41",
                hf_config=config(
                    architectures=["DeepseekV41ForCausalLM"],
                    model_type="deepseek_v41",
                ),
            )
        )

    def test_deepseek_model_with_glm_parser_is_flagged(self):
        message = detect_tool_parser_mismatch(
            "models/DeepSeek-V4.1-Flash-1017",
            "glm47",
            hf_config=config(
                architectures=["DeepseekV41ForCausalLM"],
                model_type="deepseek_v41",
            ),
        )
        self.assertIsNotNone(message)
        self.assertIn("'glm47'", message)
        self.assertIn("deepseekv41", message)

    def test_parser_underscore_and_hyphen_variants_are_recognized(self):
        hf_config = config(model_type="deepseek_v41")
        self.assertIsNone(
            detect_tool_parser_mismatch(
                "ds-model", "deepseek_v41", hf_config=hf_config
            )
        )
        self.assertIsNone(
            detect_tool_parser_mismatch(
                "ds-model", "DeepSeek-V41", hf_config=hf_config
            )
        )

    def test_nested_text_config_model_type_is_used(self):
        # DeepSeek-V4.1 configs nest the text model under text_config.
        nested = config(
            architectures=["DeepseekV41ForConditionalGeneration"],
            model_type="deepseek_v41",
            text_model_type="deepseek_v41_text",
        )
        self.assertIsNotNone(
            detect_tool_parser_mismatch(
                "ds-model", "glm47", hf_config=nested
            )
        )

    def test_path_fallback_when_config_omitted(self):
        self.assertIsNotNone(
            detect_tool_parser_mismatch("/models/GLM-5.3/", "deepseekv4")
        )
        self.assertIsNone(
            detect_tool_parser_mismatch("/models/Qwen3-32B", "qwen25")
        )

    def test_generic_or_unknown_parser_is_not_flagged(self):
        glm_config = config(model_type="glm_moe_dsa")
        # Portable parsers may legitimately serve many families.
        self.assertIsNone(
            detect_tool_parser_mismatch(
                "glm-model", "hermes", hf_config=glm_config
            )
        )
        self.assertIsNone(
            detect_tool_parser_mismatch(
                "glm-model", "response_template", hf_config=glm_config
            )
        )
        # Unknown parser name: never second-guessed.
        self.assertIsNone(
            detect_tool_parser_mismatch(
                "glm-model", "some_future_parser", hf_config=glm_config
            )
        )

    def test_missing_parser_or_unidentifiable_model_is_silent(self):
        self.assertIsNone(detect_tool_parser_mismatch("glm-model", None))
        self.assertIsNone(detect_tool_parser_mismatch(None, "glm47"))
        self.assertIsNone(
            detect_tool_parser_mismatch(
                "internal-model-abc123",
                "glm47",
                hf_config=config(architectures=["UnregisteredCustomForCausalLM"]),
            )
        )

    def test_config_identity_takes_precedence_over_path(self):
        # Path says GLM but the loaded config identifies a Qwen model; the
        # config wins, and a GLM parser mismatches.
        message = detect_tool_parser_mismatch(
            "/cache/glm-alias-folder",
            "glm47",
            hf_config=config(
                architectures=["Qwen3ForCausalLM"], model_type="qwen3"
            ),
        )
        self.assertIsNotNone(message)
        self.assertIn("qwen", message)

    def test_log_helper_emits_warning_and_returns_message(self):
        with self.assertLogs(
            "kv_capacity_estimator.tokenize.parser_matching", level="WARNING"
        ) as captured:
            message = log_tool_parser_mismatch("models/GLM-5.3", "deepseekv41")
        self.assertIsNotNone(message)
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), message)

    def test_log_helper_is_silent_for_matching_pair(self):
        # No records are emitted, so assertNoLogs is the strict check.
        with self.assertNoLogs(
            "kv_capacity_estimator.tokenize.parser_matching", level="WARNING"
        ):
            self.assertIsNone(
                log_tool_parser_mismatch("models/GLM-5.3", "glm47")
            )


if __name__ == "__main__":
    unittest.main()
