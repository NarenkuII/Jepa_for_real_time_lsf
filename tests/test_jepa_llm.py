from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.models.jepa_llm import JepaLlmPrefix, bidirectional_alignment_loss


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
    assert torch.isfinite(output.generation_loss)
    assert torch.isfinite(output.alignment_loss)
    assert -1.0 <= float(output.alignment_cosine) <= 1.0
    output.loss.backward()
    assert llm.embedding.weight.grad is None
    assert model.projector[-1].weight.grad is not None
    state = model.adapter_state_dict()
    assert state["llm_trainable"] == {}
    model.load_adapter_state_dict(state)
    generated = model.greedy_generate(keypoints, skeleton_mask, 1, 2, max_new_tokens=3)
    assert generated.shape[0] == 2
    assert generated.shape[1] <= 4


def test_alignment_loss_supports_single_sample():
    visual = torch.tensor([[1.0, 0.0]], requires_grad=True)
    text = torch.tensor([[0.0, 1.0]])
    loss = bidirectional_alignment_loss(visual, text)
    assert loss > 0
    loss.backward()
    assert visual.grad is not None


def test_adapter_checkpoint_restores_trainable_llm_weights():
    llm = TinyCausalLM()
    model = JepaLlmPrefix(TinyEncoder(), 16, llm, prefix_tokens=4, resampler_heads=4)
    model.llm.output.requires_grad_(True)
    state = model.adapter_state_dict()
    expected = state["llm_trainable"]["output.weight"].clone()
    with torch.no_grad():
        model.llm.output.weight.zero_()
    model.load_adapter_state_dict(state)
    assert torch.equal(model.llm.output.weight, expected)
