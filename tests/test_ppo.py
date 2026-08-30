from catanlab.ppo import (
    PPOTransition,
    PPORolloutBuffer,
)


def make_transition(
    action_id=0,
):
    return PPOTransition(
        observation=(
            0.0,
            1.0,
        ),
        legal_mask=(
            True,
            False,
        ),
        action_id=action_id,
        log_prob=-0.5,
        value=0.25,
    )


def test_rollout_buffer_appends():
    buffer = PPORolloutBuffer()

    buffer.append(
        make_transition()
    )

    assert len(buffer) == 1


def test_finish_episode_sets_terminal_reward():
    buffer = PPORolloutBuffer()

    buffer.append(
        make_transition()
    )

    buffer.append(
        make_transition()
    )

    buffer.finish_episode(
        terminal_reward=1.0
    )

    assert (
        buffer.transitions[0].reward
        == 0.0
    )

    assert (
        buffer.transitions[0].done
        is False
    )

    assert (
        buffer.transitions[1].reward
        == 1.0
    )

    assert (
        buffer.transitions[1].done
        is True
    )


def test_rollout_tensor_shapes():
    buffer = PPORolloutBuffer()

    buffer.append(
        make_transition()
    )

    buffer.append(
        make_transition()
    )

    tensors = buffer.to_tensors()

    assert (
        tensors[
            "observations"
        ].shape
        == (2, 2)
    )

    assert (
        tensors[
            "legal_masks"
        ].shape
        == (2, 2)
    )

    assert (
        tensors[
            "action_ids"
        ].shape
        == (2,)
    )


def test_ppo_agent_records_real_turn():
    from catanlab.board import (
        build_random_board,
    )
    from catanlab.devcards import (
        build_dev_card_deck,
    )
    from catanlab.economy import (
        PlayerInventory,
        ResourceBank,
    )
    from catanlab.rl_agent import (
        PPORolloutAgent,
    )
    from catanlab.rl_model import (
        CatanActorCritic,
    )
    from catanlab.simulation import (
        PlayerState,
    )
    from catanlab.strategies import (
        StrategyType,
    )
    from catanlab.turns import (
        AdaptiveStrategyAgent,
        run_turn,
    )

    board = build_random_board(
        seed=123
    )

    players = [
        PlayerState(
            player_id=i
        )
        for i in range(4)
    ]

    inventories = [
        PlayerInventory()
        for _ in range(4)
    ]

    deck = build_dev_card_deck(
        seed=456
    )

    bank = ResourceBank()

    ppo_agent = PPORolloutAgent(
        StrategyType.FIVE_RESOURCE,
        model=CatanActorCritic(),
        deterministic=False,
        seed=1,
    )

    agents = [
        ppo_agent,
        AdaptiveStrategyAgent(
            StrategyType.FIVE_RESOURCE
        ),
        AdaptiveStrategyAgent(
            StrategyType.FIVE_RESOURCE
        ),
        AdaptiveStrategyAgent(
            StrategyType.FIVE_RESOURCE
        ),
    ]

    result = run_turn(
        board,
        players,
        inventories,
        agents,
        player_id=0,
        roll=6,
        dev_deck=deck,
        bank=bank,
    )

    assert result is not None
    assert len(
        ppo_agent.rollout
    ) >= 1

    transition = (
        ppo_agent
        .rollout
        .transitions[0]
    )

    assert (
        len(
            transition.observation
        )
        == 1138
    )

    assert (
        len(
            transition.legal_mask
        )
        == 202
    )

    assert transition.legal_mask[
        transition.action_id
    ]


def test_compute_gae_terminal_win():
    import torch

    from catanlab.ppo import (
        compute_gae,
    )

    rewards = torch.tensor(
        [0.0, 0.0, 1.0]
    )

    dones = torch.tensor(
        [False, False, True]
    )

    values = torch.zeros(3)

    advantages, returns = (
        compute_gae(
            rewards,
            dones,
            values,
            gamma=1.0,
            gae_lambda=1.0,
        )
    )

    expected = torch.tensor(
        [1.0, 1.0, 1.0]
    )

    assert torch.allclose(
        advantages,
        expected,
    )

    assert torch.allclose(
        returns,
        expected,
    )


def test_compute_gae_terminal_loss():
    import torch

    from catanlab.ppo import (
        compute_gae,
    )

    rewards = torch.zeros(3)

    dones = torch.tensor(
        [False, False, True]
    )

    values = torch.zeros(3)

    advantages, returns = (
        compute_gae(
            rewards,
            dones,
            values,
        )
    )

    assert torch.allclose(
        advantages,
        torch.zeros(3),
    )

    assert torch.allclose(
        returns,
        torch.zeros(3),
    )


def test_ppo_update_changes_parameters():
    import torch

    from catanlab.ppo import (
        compute_gae,
        ppo_update,
    )
    from catanlab.rl_model import (
        CatanActorCritic,
        mask_policy_logits,
    )

    torch.manual_seed(123)

    model = CatanActorCritic()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    observations = torch.randn(
        8,
        1138,
    )

    legal_masks = torch.ones(
        8,
        202,
        dtype=torch.bool,
    )

    with torch.no_grad():
        logits, values = model(
            observations
        )

        masked_logits = (
            mask_policy_logits(
                logits,
                legal_masks,
            )
        )

        distribution = (
            torch.distributions.Categorical(
                logits=masked_logits
            )
        )

        action_ids = (
            distribution.sample()
        )

        old_log_probs = (
            distribution.log_prob(
                action_ids
            )
        )

    rewards = torch.zeros(8)
    rewards[-1] = 1.0

    dones = torch.zeros(
        8,
        dtype=torch.bool,
    )

    dones[-1] = True

    advantages, returns = (
        compute_gae(
            rewards,
            dones,
            values.squeeze(-1),
        )
    )

    before = [
        parameter.detach().clone()
        for parameter
        in model.parameters()
    ]

    metrics = ppo_update(
        model,
        optimizer,
        observations,
        legal_masks,
        action_ids,
        old_log_probs,
        advantages,
        returns,
        epochs=2,
        batch_size=4,
    )

    after = list(
        model.parameters()
    )

    assert any(
        not torch.equal(
            old,
            new.detach(),
        )
        for old, new
        in zip(before, after)
    )

    assert (
        metrics.value_loss >= 0.0
    )

    assert (
        metrics.clip_fraction
        >= 0.0
    )

    assert (
        metrics.clip_fraction
        <= 1.0
    )


def test_gae_is_computed_per_episode():
    import torch

    from catanlab.ppo import (
        compute_gae,
    )

    win_rewards = torch.tensor(
        [0.0, 1.0]
    )

    win_dones = torch.tensor(
        [False, True]
    )

    loss_rewards = torch.tensor(
        [0.0, 0.0]
    )

    loss_dones = torch.tensor(
        [False, True]
    )

    values = torch.zeros(2)

    win_advantages, _ = compute_gae(
        win_rewards,
        win_dones,
        values,
        gamma=1.0,
        gae_lambda=1.0,
    )

    loss_advantages, _ = compute_gae(
        loss_rewards,
        loss_dones,
        values,
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert torch.allclose(
        win_advantages,
        torch.ones(2),
    )

    assert torch.allclose(
        loss_advantages,
        torch.zeros(2),
    )


def test_public_vp_margin_potential():
    from catanlab.ppo import (
        public_vp_margin_potential,
    )
    from catanlab.simulation import (
        PlayerState,
    )

    players = [
        PlayerState(
            player_id=0,
            settlements=[0, 1],
            cities=[2],
        ),
        PlayerState(
            player_id=1,
            settlements=[3],
            cities=[4, 5],
        ),
    ]

    # Player 0 public VP:
    # 2 settlements + one city*2 = 4
    #
    # Player 1 public VP:
    # 1 settlement + two cities*2 = 5
    #
    # margin = (4 - 5) / 10 = -0.1
    result = (
        public_vp_margin_potential(
            players,
            0,
        )
    )

    assert abs(
        result - (-0.1)
    ) < 1e-9


def test_public_vp_potential_ignores_hidden_vp():
    from catanlab.ppo import (
        public_vp_margin_potential,
    )
    from catanlab.simulation import (
        PlayerState,
    )

    players = [
        PlayerState(
            player_id=0,
            settlements=[0],
        ),
        PlayerState(
            player_id=1,
            settlements=[1],
        ),
    ]

    before = (
        public_vp_margin_potential(
            players,
            0,
        )
    )

    players[1].dev_cards.append(
        "victory_point"
    )

    after = (
        public_vp_margin_potential(
            players,
            0,
        )
    )

    assert before == after


def test_potential_shaping_rewards():
    from catanlab.ppo import (
        PPOTransition,
        PPORolloutBuffer,
    )

    buffer = PPORolloutBuffer()

    buffer.append(
        PPOTransition(
            observation=(0.0,),
            legal_mask=(True,),
            action_id=0,
            log_prob=0.0,
            value=0.0,
            potential=0.0,
        )
    )

    buffer.append(
        PPOTransition(
            observation=(0.0,),
            legal_mask=(True,),
            action_id=0,
            log_prob=0.0,
            value=0.0,
            potential=0.1,
        )
    )

    buffer.finish_episode_with_potential(
        terminal_reward=1.0,
        terminal_potential=0.3,
        shaping_alpha=0.5,
        gamma=1.0,
    )

    # First:
    # 0.5 * (0.1 - 0.0) = 0.05
    assert abs(
        buffer.transitions[0].reward
        - 0.05
    ) < 1e-9

    # Last:
    # terminal 1.0
    # + 0.5 * (0.3 - 0.1)
    # = 1.1
    assert abs(
        buffer.transitions[1].reward
        - 1.1
    ) < 1e-9

    assert (
        buffer.transitions[0].done
        is False
    )

    assert (
        buffer.transitions[1].done
        is True
    )


def test_potential_shaping_telescopes():
    from catanlab.ppo import (
        PPOTransition,
        PPORolloutBuffer,
    )

    buffer = PPORolloutBuffer()

    for potential in (
        -0.2,
        -0.1,
        0.2,
    ):
        buffer.append(
            PPOTransition(
                observation=(0.0,),
                legal_mask=(True,),
                action_id=0,
                log_prob=0.0,
                value=0.0,
                potential=potential,
            )
        )

    buffer.finish_episode_with_potential(
        terminal_reward=1.0,
        terminal_potential=0.4,
        shaping_alpha=0.25,
        gamma=1.0,
    )

    total_reward = sum(
        transition.reward
        for transition
        in buffer.transitions
    )

    # Sum of shaping terms must telescope:
    #
    # 1 + alpha * (Phi_terminal - Phi_initial)
    expected = (
        1.0
        + 0.25
        * (
            0.4 - (-0.2)
        )
    )

    assert abs(
        total_reward - expected
    ) < 1e-9


def test_discounted_potential_shaping_telescopes():
    from catanlab.ppo import (
        PPOTransition,
        PPORolloutBuffer,
    )

    gamma = 0.99
    alpha = 0.25

    potentials = (
        -0.2,
        -0.1,
        0.2,
    )

    buffer = PPORolloutBuffer()

    for potential in potentials:
        buffer.append(
            PPOTransition(
                observation=(0.0,),
                legal_mask=(True,),
                action_id=0,
                log_prob=0.0,
                value=0.0,
                potential=potential,
            )
        )

    buffer.finish_episode_with_potential(
        terminal_reward=0.0,
        terminal_potential=0.0,
        shaping_alpha=alpha,
        gamma=gamma,
    )

    discounted_shaping = sum(
        (
            gamma ** index
        )
        * transition.reward
        for index, transition
        in enumerate(
            buffer.transitions
        )
    )

    # With an absorbing terminal potential of zero:
    #
    # sum gamma^t F_t
    # = -alpha * Phi(s_0)
    expected = (
        -alpha
        * potentials[0]
    )

    assert abs(
        discounted_shaping
        - expected
    ) < 1e-8


def test_ppo_positive_bc_kl_requires_reference():
    import torch

    from catanlab.ppo import ppo_update
    from catanlab.rl_model import CatanActorCritic

    model = CatanActorCritic(
        observation_dim=4,
        action_dim=3,
        hidden_dim=8,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    observations = torch.zeros(
        (4, 4),
        dtype=torch.float32,
    )

    legal_masks = torch.ones(
        (4, 3),
        dtype=torch.bool,
    )

    action_ids = torch.zeros(
        4,
        dtype=torch.long,
    )

    old_log_probs = torch.zeros(
        4,
        dtype=torch.float32,
    )

    advantages = torch.ones(
        4,
        dtype=torch.float32,
    )

    returns = torch.zeros(
        4,
        dtype=torch.float32,
    )

    try:
        ppo_update(
            model,
            optimizer,
            observations,
            legal_masks,
            action_ids,
            old_log_probs,
            advantages,
            returns,
            epochs=1,
            batch_size=4,
            bc_kl_coefficient=0.01,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Positive BC KL coefficient should "
            "require a reference model."
        )


def test_ppo_reports_zero_bc_kl_at_identical_policy_start():
    import copy
    import torch

    from catanlab.ppo import ppo_update
    from catanlab.rl_model import CatanActorCritic

    torch.manual_seed(0)

    model = CatanActorCritic(
        observation_dim=4,
        action_dim=3,
        hidden_dim=8,
    )

    reference = copy.deepcopy(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0,
    )

    observations = torch.randn(
        (8, 4),
    )

    legal_masks = torch.ones(
        (8, 3),
        dtype=torch.bool,
    )

    with torch.no_grad():
        logits, _ = model(
            observations
        )

        distribution = (
            torch.distributions.Categorical(
                logits=logits
            )
        )

        action_ids = distribution.sample()

        old_log_probs = (
            distribution.log_prob(
                action_ids
            )
        )

    advantages = torch.linspace(
        -1.0,
        1.0,
        8,
    )

    returns = torch.zeros(
        8,
    )

    metrics = ppo_update(
        model,
        optimizer,
        observations,
        legal_masks,
        action_ids,
        old_log_probs,
        advantages,
        returns,
        epochs=1,
        batch_size=8,
        reference_model=reference,
        bc_kl_coefficient=0.01,
    )

    assert metrics.bc_kl >= 0.0
    assert metrics.bc_kl < 1e-7


def test_ppo_bc_kl_is_finite_with_masked_actions():
    import copy
    import math

    import torch

    from catanlab.ppo import ppo_update
    from catanlab.rl_model import CatanActorCritic

    torch.manual_seed(123)

    model = CatanActorCritic(
        observation_dim=4,
        action_dim=5,
        hidden_dim=8,
    )

    reference = copy.deepcopy(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0,
    )

    observations = torch.randn(
        (8, 4),
        dtype=torch.float32,
    )

    # Every row has legal and illegal actions.
    legal_masks = torch.tensor(
        [
            [True,  True,  False, False, False],
            [True,  False, True,  False, False],
            [False, True,  True,  False, False],
            [True,  False, False, True,  False],
            [False, True,  False, True,  False],
            [True,  False, False, False, True ],
            [False, True,  False, False, True ],
            [False, False, True,  True,  False],
        ],
        dtype=torch.bool,
    )

    with torch.no_grad():
        logits, _ = model(
            observations
        )

        masked_logits = logits.masked_fill(
            ~legal_masks,
            float("-inf"),
        )

        distribution = (
            torch.distributions.Categorical(
                logits=masked_logits
            )
        )

        action_ids = distribution.sample()

        old_log_probs = (
            distribution.log_prob(
                action_ids
            )
        )

    advantages = torch.linspace(
        -1.0,
        1.0,
        8,
    )

    returns = torch.zeros(
        8,
        dtype=torch.float32,
    )

    metrics = ppo_update(
        model,
        optimizer,
        observations,
        legal_masks,
        action_ids,
        old_log_probs,
        advantages,
        returns,
        epochs=1,
        batch_size=8,
        reference_model=reference,
        bc_kl_coefficient=0.01,
    )

    assert math.isfinite(
        metrics.bc_kl
    )

    assert math.isfinite(
        metrics.total_loss
    )

    # Identical current/reference policies should
    # begin with essentially zero anchor KL.
    assert metrics.bc_kl < 1e-7
