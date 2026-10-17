import os
from types import SimpleNamespace

import pytest
import torch
from helpers import TinyTokenizer

from models.dream import DREAM, DreamGenerationConfig, sample_tokens
from models.llada import LLADA, add_gumbel_noise


class StepModel:
    device = torch.device(os.environ.get("DINGO_TEST_DEVICE", "cpu"))

    def __init__(self, preferences):
        self.preferences = preferences
        self.inputs = []

    def __call__(self, x, *args):
        self.inputs.append(x.clone())
        preference = self.preferences[min(len(self.inputs) - 1, len(self.preferences) - 1)]
        logits = torch.full((*x.shape, 4), -10.0, device=x.device)
        logits[:, :, preference] = 10.0
        return SimpleNamespace(logits=logits)


def run_loop(
    backend,
    model,
    store,
    *,
    steps=2,
    gen_length=1,
    block_length=1,
    constrain_at=0,
    alg="entropy",
    mode="diffusion_constrained",
    **kwargs,
):
    generator = backend(
        model,
        TinyTokenizer(),
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        constrain_at=constrain_at,
        **kwargs,
    )
    generator.dfa_store = store
    for name in [
        "edge_src",
        "edge_dst",
        "edge_tok",
        "edge_src_nomdm",
        "edge_dst_nomdm",
        "edge_tok_nomdm",
    ]:
        setattr(generator, name, getattr(store, name))
    # The published LLaDA mask ID is replaceable for this tiny model fixture.
    if backend is LLADA:
        generator.mask_id = 3
        return generator.generate(
            torch.tensor([[0]], device=model.device), constraint_mode=mode
        ), model
    config = DreamGenerationConfig(max_length=1 + gen_length, steps=steps, mask_token_id=3, alg=alg)
    return generator.constrained_sample(
        torch.tensor([[0]], device=model.device),
        None,
        config,
        lambda i, x, logits: x,
        lambda i, x, logits: logits,
        constraint_mode=mode,
    ), model


@pytest.mark.parametrize("backend", [LLADA, DREAM])
@pytest.mark.parametrize("opportunistic", [False, True])
def test_each_step_uses_current_logits_for_revealed_positions(
    backend, opportunistic, store_factory
):
    store = store_factory("a|b", opportunistic=opportunistic)
    output, _ = run_loop(backend, StepModel([0, 1]), store)
    assert output[:, 1:].tolist() == [[1]]


@pytest.mark.parametrize("backend", [LLADA, DREAM])
def test_state_is_carried_between_blocks(backend, store_factory):
    store = store_factory("ab")
    output, _ = run_loop(backend, StepModel([0, 1]), store, gen_length=2)
    assert output[:, 1:].tolist() == [[0, 1]]


@pytest.mark.parametrize("backend", [LLADA, DREAM])
def test_partial_unmasking_has_a_real_token_completion(backend, store_factory):
    store = store_factory("ab")
    output, model = run_loop(backend, StepModel([1]), store, gen_length=2, block_length=2)
    for x in model.inputs[1:]:
        assert store.check_is_reachable(x[0, 1:].tolist(), is_final_block=True)
    assert output[:, 1:].tolist() == [[0, 1]]


def test_dream_origin_schedule_decodes_a_vector_not_a_vocab_matrix(store_factory):
    output, _ = run_loop(DREAM, StepModel([0]), store_factory("a"), steps=1, alg="origin")
    assert output[:, 1:].tolist() == [[0]]


def test_margin_confidence_uses_last_vocabulary_axis():
    scores = torch.tensor([[[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]])
    conf, tokens = sample_tokens(scores, margin_confidence=True)
    assert conf.shape == (1, 2)
    torch.testing.assert_close(conf, torch.tensor([[0.420512, 0.420512]]), atol=1e-5, rtol=1e-5)


def test_gumbel_scores_are_stable_for_extreme_logits():
    scores = add_gumbel_noise(torch.tensor([[1000.0, 999.0, -1000.0]]), 0.5)
    assert torch.isfinite(scores).all()


@pytest.mark.parametrize("backend", [LLADA, DREAM])
def test_invalid_constrain_threshold_is_rejected(backend, store_factory):
    with pytest.raises(ValueError, match="constrain_at"):
        run_loop(backend, StepModel([0]), store_factory("a"), constrain_at=2)


@pytest.mark.parametrize("backend", [LLADA, DREAM])
def test_forward_conditions_only_on_prefix_and_active_block(backend, store_factory):
    _, model = run_loop(backend, StepModel([0, 1]), store_factory("ab"), gen_length=2)
    assert [x.shape[1] for x in model.inputs] == [2, 3]


@pytest.mark.parametrize("filter_args", [{"top_k": 1}, {"top_p": 0.5}])
def test_dream_constraints_respect_filtered_probability_support(store_factory, filter_args):
    generator = DREAM(StepModel([0]), TinyTokenizer(), steps=1, gen_length=1, block_length=1)
    generator.dfa_store = store_factory("b")
    config = DreamGenerationConfig(
        max_length=2, steps=1, mask_token_id=3, alg="topk", **filter_args
    )
    with pytest.raises(ValueError, match="supported constrained path"):
        generator.constrained_sample(
            torch.tensor([[0]], device=generator.model.device),
            None,
            config,
            lambda i, x, logits: x,
            lambda i, x, logits: logits,
            constraint_mode="diffusion_constrained",
        )


@pytest.mark.parametrize("backend", [LLADA, DREAM])
def test_interrupted_request_does_not_poison_next_generation(backend, store_factory):
    store = store_factory("ab")

    class InterruptedModel(StepModel):
        def __call__(self, x, *args):
            if self.inputs:
                raise RuntimeError("model interrupted")
            return super().__call__(x, *args)

    # One token cannot satisfy 'ab'; the first deferred decode fails, then
    # an unrelated model exception interrupts before the validation boundary.
    with pytest.raises(RuntimeError, match="model interrupted"):
        run_loop(backend, InterruptedModel([0]), store)
    output, _ = run_loop(backend, StepModel([0, 1]), store, gen_length=2, block_length=2)
    assert output[:, 1:].tolist() == [[0, 1]]


def test_dream_remasking_ties_preserve_original_full_context_topk_shape(store_factory):
    # PyTorch breaks equal-confidence ties by index/shape. Preserve the old
    # prompt+active-block shape so removing synchronization does not alter it.
    output, model = run_loop(
        DREAM,
        StepModel([0]),
        store_factory("a*"),
        steps=3,
        gen_length=8,
        block_length=8,
        mode="unconstrained",
        alg="topk",
    )
    reference = torch.tensor([-torch.inf] + [1.0] * 8, device=model.device).topk(3).indices
    expected = torch.full((9,), 3, dtype=torch.long, device=model.device)
    expected[0] = 0
    expected[reference] = 0
    torch.testing.assert_close(model.inputs[1][0], expected)
