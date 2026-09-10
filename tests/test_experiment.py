import json

from disc.core import DetectionResult
from disc.experiment import ExperimentConfig, build_record, checkpoint_path, load_prompts, parse_key


def test_parse_key_keeps_disc_key_types():
    assert parse_key("0x2a") == 42
    assert parse_key("42") == 42
    assert parse_key("research-key") == "research-key"


def test_load_prompts_supports_mirrormark_json_and_lines(tmp_path):
    json_file = tmp_path / "prompts.json"
    json_file.write_text(json.dumps(["one", "two"]), encoding="utf-8")
    line_file = tmp_path / "prompts.txt"
    line_file.write_text("one\n\ntwo\n", encoding="utf-8")
    assert load_prompts("fallback", str(json_file)) == ["one", "two"]
    assert load_prompts("fallback", str(line_file)) == ["one", "two"]


def test_checkpoint_record_uses_shared_mirrormark_fields(tmp_path):
    config = ExperimentConfig(2, 3, "token_ngram", 4, False, "direct", True, 0.01, "int")
    result = DetectionResult(True, 5, 12, 0.002, 0.006, 18.0, 9, (1, 1, 1), (0.1, 0.2, 0.3))
    row = build_record(index=7, prompt="p", response="r", token_ids=[1, 2], checkpoint=2,
                       perplexity=3.0, generation_seconds=1.2, decoding_seconds=0.3,
                       result=result, config=config, payload=5)
    assert {"idx", "prompt", "response", "checkpoint_tokens", "ppl", "score", "z", "pvalue", "time", "step_stats"} <= row.keys()
    assert row["message"] == 5 and row["decoded_msg"] == 5 and row["message_correct"]
    assert checkpoint_path(tmp_path, "watermark", 2, 3, 100).name == "watermark_m2_pos3_100tokens.jsonl"
