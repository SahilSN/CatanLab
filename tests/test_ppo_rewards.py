from catanlab.ppo import (
    terminal_reward,
)


def test_win_reward():
    assert (
        terminal_reward(
            won=True,
            final_vp=10.0,
            reward_mode="win",
        )
        == 1.0
    )

    assert (
        terminal_reward(
            won=False,
            final_vp=8.0,
            reward_mode="win",
        )
        == 0.0
    )


def test_vp_reward():
    assert (
        terminal_reward(
            won=False,
            final_vp=8.0,
            reward_mode="vp",
        )
        == 0.8
    )

    assert (
        terminal_reward(
            won=True,
            final_vp=10.0,
            reward_mode="vp",
        )
        == 1.0
    )


def test_vp_win_reward():
    assert (
        terminal_reward(
            won=False,
            final_vp=8.0,
            reward_mode="vp_win",
        )
        == 0.8
    )

    assert (
        terminal_reward(
            won=True,
            final_vp=10.0,
            reward_mode="vp_win",
        )
        == 2.0
    )


def test_unknown_reward_mode_rejected():
    try:
        terminal_reward(
            won=False,
            final_vp=5.0,
            reward_mode="unknown",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected invalid reward mode "
            "to raise ValueError."
        )


def test_win_margin_loss_far_behind():
    reward = terminal_reward(
        won=False,
        final_vp=5.0,
        best_opponent_vp=10.0,
        reward_mode="win_margin",
    )

    assert abs(
        reward - (-0.05)
    ) < 1e-9


def test_win_margin_close_loss():
    reward = terminal_reward(
        won=False,
        final_vp=9.0,
        best_opponent_vp=10.0,
        reward_mode="win_margin",
    )

    assert abs(
        reward - (-0.01)
    ) < 1e-9


def test_win_margin_win():
    reward = terminal_reward(
        won=True,
        final_vp=10.0,
        best_opponent_vp=8.0,
        reward_mode="win_margin",
    )

    assert abs(
        reward - 1.02
    ) < 1e-9


def test_win_margin_requires_opponent_vp():
    try:
        terminal_reward(
            won=False,
            final_vp=8.0,
            reward_mode="win_margin",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected win_margin without "
            "best_opponent_vp to fail."
        )
