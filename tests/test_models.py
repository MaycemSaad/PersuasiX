"""Tests for model architectures."""

import torch
import pytest

from src.models.detector import PersuasionDetector


class TestPersuasionDetector:
    @pytest.fixture
    def model(self):
        return PersuasionDetector(
            model_name="roberta-base",
            num_labels=18,
            dropout=0.1,
        )

    @pytest.fixture
    def dummy_input(self):
        return {
            "input_ids": torch.randint(0, 1000, (2, 64)),
            "attention_mask": torch.ones(2, 64, dtype=torch.long),
        }

    @pytest.fixture
    def dummy_labels(self):
        return torch.zeros(2, 18)

    def test_forward_shape(self, model, dummy_input):
        output = model(**dummy_input)
        assert "logits" in output
        assert output["logits"].shape == (2, 18)

    def test_forward_with_labels(self, model, dummy_input, dummy_labels):
        output = model(**dummy_input, labels=dummy_labels)
        assert "loss" in output
        assert output["loss"].ndim == 0

    def test_predict(self, model, dummy_input):
        result = model.predict(**dummy_input, threshold=0.5)
        assert "probabilities" in result
        assert "predictions" in result
        assert result["probabilities"].shape == (2, 18)
        assert result["predictions"].dtype == torch.long

    def test_probabilities_range(self, model, dummy_input):
        result = model.predict(**dummy_input)
        probs = result["probabilities"]
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_save_and_load(self, model, tmp_path):
        path = str(tmp_path / "model.pt")
        model.save_pretrained(path)
        loaded = PersuasionDetector.from_pretrained(path)
        assert loaded.num_labels == model.num_labels

    def test_freeze_layers(self):
        model = PersuasionDetector(freeze_encoder_layers=4)
        frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        assert frozen > 0

    def test_get_tokenizer(self):
        tokenizer = PersuasionDetector.get_tokenizer("roberta-base")
        tokens = tokenizer("test", return_tensors="pt")
        assert "input_ids" in tokens
