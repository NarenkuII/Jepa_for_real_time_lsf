from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.models.jepa_llm import JepaLlmPrefix


class TinyEncoder(torch.nn.Module):
    def __init__(self, in_features=6, d_model=16):
        super().__init__()
        self.projection = torch.nn.Linear(in_features, d_model)

    def forward(self, x, padding_mask):
        return self.projection(x.mean(dim=2))


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size=20, d_model=24):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.output = torch.nn.Linear(d_model, vocab_size)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask=None, labels=None):
        logits = self.output(inputs_embeds)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)


def test_jepa_llm_prefix_forward_and_generate():
    llm = TinyCausalLM()
    model = JepaLlmPrefix(TinyEncoder(), 16, llm, prefix_tokens=4, resampler_heads=4)
    keypoints = torch.randn(2, 12, 8, 6)
    skeleton_mask = torch.ones(2, 12, dtype=torch.bool)
    skeleton_mask[1, 9:] = False
    input_ids = torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2]])
    text_mask = torch.ones_like(input_ids, dtype=torch.bool)
    output = model(keypoints, skeleton_mask, input_ids, text_mask)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert llm.embedding.weight.grad is None
    generated = model.greedy_generate(keypoints, skeleton_mask, 1, 2, max_new_tokens=3)
    assert generated.shape[0] == 2
    assert generated.shape[1] <= 4
