from typing import Tuple, Optional

import chex
import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class ReachabilityState:
    flood_input: chex.Array
    flood_count: chex.Array
    target_mask: chex.Array
    done: chex.Array


@struct.dataclass
class LayoutValidationResult:
    pot_count: chex.Array
    onion_count: chex.Array
    plate_count: chex.Array
    goal_count: chex.Array
    count_ok: chex.Array
    valid_1: chex.Array   # pot_count <= max_count
    valid_2: chex.Array   # onion_count <= max_count
    valid_3: chex.Array   # plate_count <= max_count
    valid_4: chex.Array   # goal_count <= max_count
    valid_12: chex.Array
    valid_13: chex.Array
    valid_14: chex.Array
    valid_23: chex.Array
    valid_24: chex.Array
    valid_34: chex.Array
    valid_123: chex.Array
    valid_124: chex.Array
    valid_134: chex.Array
    valid_234: chex.Array
    valid_all: chex.Array  # pot+onion+plate+goal all <= max_count
    onion_reachable: chex.Array
    plate_reachable: chex.Array
    pot_reachable: chex.Array
    goal_reachable: chex.Array
    reachable_ok: chex.Array


def make_tile_mask(env_map: chex.Array, tile_idx: int) -> chex.Array:
    return env_map == tile_idx


def make_multi_tile_mask(env_map: chex.Array, tile_indices: Tuple[int, ...]) -> chex.Array:
    mask = jnp.zeros(env_map.shape, dtype=bool)
    for tile_idx in tile_indices:
        mask = jnp.logical_or(mask, env_map == tile_idx)
    return mask


def shift_up(mask: chex.Array) -> chex.Array:
    shifted = jnp.zeros_like(mask)
    shifted = shifted.at[:-1, :].set(mask[1:, :])
    return shifted


def shift_down(mask: chex.Array) -> chex.Array:
    shifted = jnp.zeros_like(mask)
    shifted = shifted.at[1:, :].set(mask[:-1, :])
    return shifted


def shift_left(mask: chex.Array) -> chex.Array:
    shifted = jnp.zeros_like(mask)
    shifted = shifted.at[:, :-1].set(mask[:, 1:])
    return shifted


def shift_right(mask: chex.Array) -> chex.Array:
    shifted = jnp.zeros_like(mask)
    shifted = shifted.at[:, 1:].set(mask[:, :-1])
    return shifted


def _get_existing_tile_indices(object_to_index: dict, candidate_keys: Tuple[str, ...]) -> Tuple[int, ...]:
    vals = []
    for k in candidate_keys:
        if k in object_to_index:
            vals.append(object_to_index[k])
    return tuple(vals)


def infer_player_tiles(object_to_index: dict) -> Tuple[int, ...]:
    return _get_existing_tile_indices(
        object_to_index,
        ("player", "player2", "agent", "agent1", "agent2"),
    )


def infer_blocking_tiles(object_to_index: dict) -> Tuple[int, ...]:
    """
    플레이어가 '올라설 수 없는' 타일들을 모은다.
    없는 key는 자동 무시.
    """
    return _get_existing_tile_indices(
        object_to_index,
        (
            "wall",
            "counter",
            "pot",
            "onion_pile",
            "plate_pile",
            "goal",
        ),
    )


def make_passable_mask(env_map: chex.Array, object_to_index: dict) -> chex.Array:
    blocking_tiles = infer_blocking_tiles(object_to_index)

    if len(blocking_tiles) == 0:
        # blocking 정보를 하나도 모르겠으면 전 칸 passable 취급
        return jnp.ones(env_map.shape, dtype=bool)

    blocked_mask = make_multi_tile_mask(env_map, blocking_tiles)
    passable_mask = jnp.logical_not(blocked_mask)
    return passable_mask


def make_adjacent_access_mask(
    env_map: chex.Array,
    object_tile_idx: int,
    object_to_index: dict,
) -> chex.Array:
    """
    object 타일 자체가 아니라,
    object 상하좌우 중 플레이어가 설 수 있는 칸(passable)을 target 으로 만든다.
    """
    object_mask = make_tile_mask(env_map, object_tile_idx)
    passable_mask = make_passable_mask(env_map, object_to_index)

    adjacent_mask = shift_up(object_mask)
    adjacent_mask = jnp.logical_or(adjacent_mask, shift_down(object_mask))
    adjacent_mask = jnp.logical_or(adjacent_mask, shift_left(object_mask))
    adjacent_mask = jnp.logical_or(adjacent_mask, shift_right(object_mask))

    access_mask = jnp.logical_and(adjacent_mask, passable_mask)
    return access_mask


def is_not_done(state: ReachabilityState):
    return jnp.logical_not(state.done)


def _expand_flood_once(flood_input: chex.Array) -> chex.Array:
    """
    flood_input[..., 0] = occupied_map (1이면 막힘)
    flood_input[..., 1] = current_flood
    """
    occupied_map = flood_input[..., 0]
    current_flood = flood_input[..., 1]

    up = shift_up(current_flood)
    down = shift_down(current_flood)
    left = shift_left(current_flood)
    right = shift_right(current_flood)

    expanded = jnp.logical_or(current_flood > 0, up > 0)
    expanded = jnp.logical_or(expanded, down > 0)
    expanded = jnp.logical_or(expanded, left > 0)
    expanded = jnp.logical_or(expanded, right > 0)

    expanded = jnp.logical_and(expanded, occupied_map == 0)
    expanded = expanded.astype(jnp.float32)

    flood_out = jnp.stack([occupied_map, expanded], axis=-1)
    return flood_out


def flood_step_until_target(state: ReachabilityState) -> ReachabilityState:
    flood_input = state.flood_input
    flood_count = state.flood_count
    target_mask = state.target_mask

    flood_out = _expand_flood_once(flood_input)
    new_flood_count = flood_count + flood_out[..., -1]

    reached_target = jnp.any(jnp.logical_and(new_flood_count > 0, target_mask))
    no_change = jnp.all(flood_input == flood_out)
    done = jnp.logical_or(reached_target, no_change)

    return ReachabilityState(
        flood_input=flood_out,
        flood_count=new_flood_count,
        target_mask=target_mask,
        done=done,
    )


def is_target_reachable(
    env_map: chex.Array,
    start_mask: chex.Array,
    target_mask: chex.Array,
    object_to_index: dict,
):
    passable_mask = make_passable_mask(env_map, object_to_index)
    occupied_map = jnp.logical_not(passable_mask).astype(jnp.float32)

    start_mask = jnp.logical_and(start_mask, passable_mask)
    start_mask_f = start_mask.astype(jnp.float32)

    flood_input = jnp.stack([occupied_map, start_mask_f], axis=-1)

    init_state = ReachabilityState(
        flood_input=flood_input,
        flood_count=start_mask_f,
        target_mask=target_mask,
        done=jnp.array(False),
    )

    final_state = jax.lax.while_loop(
        cond_fun=is_not_done,
        body_fun=flood_step_until_target,
        init_val=init_state,
    )

    reachable = jnp.any(jnp.logical_and(final_state.flood_count > 0, target_mask))
    return reachable

def count_required_objects(env_map: chex.Array, object_to_index: dict, max_count: int = 2):
    pot_count = jnp.sum(env_map == object_to_index["pot"])
    onion_count = jnp.sum(env_map == object_to_index["onion_pile"])
    plate_count = jnp.sum(env_map == object_to_index["plate_pile"])
    goal_count = jnp.sum(env_map == object_to_index["goal"])

    # Baseline presence check: all required objects exist at least once.
    count_ok = (0 < pot_count) & (0 < onion_count) & (0 < plate_count) & (0 < goal_count)

    # Max-count constraints (independent from reachability / presence).
    pot_max_ok = pot_count <= max_count
    onion_max_ok = onion_count <= max_count
    plate_max_ok = plate_count <= max_count
    goal_max_ok = goal_count <= max_count

    return (
        pot_count,
        onion_count,
        plate_count,
        goal_count,
        count_ok,
        pot_max_ok,
        onion_max_ok,
        plate_max_ok,
        goal_max_ok,
    )


def validate_layout(
    maze_map: chex.Array,
    object_to_index: dict,
    max_count: int = 2,
) -> LayoutValidationResult:
    (
        pot_count,
        onion_count,
        plate_count,
        goal_count,
        count_ok,
        pot_max_ok,
        onion_max_ok,
        plate_max_ok,
        goal_max_ok,
    ) = count_required_objects(
        maze_map,
        object_to_index,
        max_count,
    )

    player_tiles = infer_player_tiles(object_to_index)
    start_mask = make_multi_tile_mask(maze_map, player_tiles) if len(player_tiles) > 0 else jnp.zeros(maze_map.shape, dtype=bool)

    onion_access_mask = make_adjacent_access_mask(
        maze_map,
        object_to_index["onion_pile"],
        object_to_index,
    )
    plate_access_mask = make_adjacent_access_mask(
        maze_map,
        object_to_index["plate_pile"],
        object_to_index,
    )
    pot_access_mask = make_adjacent_access_mask(
        maze_map,
        object_to_index["pot"],
        object_to_index,
    )
    goal_access_mask = make_adjacent_access_mask(
        maze_map,
        object_to_index["goal"],
        object_to_index,
    )

    onion_reachable = is_target_reachable(
        maze_map,
        start_mask,
        onion_access_mask,
        object_to_index,
    )
    plate_reachable = is_target_reachable(
        maze_map,
        start_mask,
        plate_access_mask,
        object_to_index,
    )
    pot_reachable = is_target_reachable(
        maze_map,
        start_mask,
        pot_access_mask,
        object_to_index,
    )
    goal_reachable = is_target_reachable(
        maze_map,
        start_mask,
        goal_access_mask,
        object_to_index,
    )

    reachable_ok = onion_reachable & plate_reachable & pot_reachable & goal_reachable

    valid_1 = count_ok & reachable_ok & pot_max_ok
    valid_2 = count_ok & reachable_ok & onion_max_ok
    valid_3 = count_ok & reachable_ok & plate_max_ok
    valid_4 = count_ok & reachable_ok & goal_max_ok

    valid_12 = count_ok & reachable_ok & pot_max_ok & onion_max_ok
    valid_13 = count_ok & reachable_ok & pot_max_ok & plate_max_ok
    valid_14 = count_ok & reachable_ok & pot_max_ok & goal_max_ok
    valid_23 = count_ok & reachable_ok & onion_max_ok & plate_max_ok
    valid_24 = count_ok & reachable_ok & onion_max_ok & goal_max_ok
    valid_34 = count_ok & reachable_ok & plate_max_ok & goal_max_ok

    valid_123 = count_ok & reachable_ok & pot_max_ok & onion_max_ok & plate_max_ok
    valid_124 = count_ok & reachable_ok & pot_max_ok & onion_max_ok & goal_max_ok
    valid_134 = count_ok & reachable_ok & pot_max_ok & plate_max_ok & goal_max_ok
    valid_234 = count_ok & reachable_ok & onion_max_ok & plate_max_ok & goal_max_ok

    valid_all = count_ok & reachable_ok & pot_max_ok & onion_max_ok & plate_max_ok & goal_max_ok

    return LayoutValidationResult(
        pot_count=pot_count,
        onion_count=onion_count,
        plate_count=plate_count,
        goal_count=goal_count,
        count_ok=count_ok,
        valid_1=valid_1,
        valid_2=valid_2,
        valid_3=valid_3,
        valid_4=valid_4,
        valid_12=valid_12,
        valid_13=valid_13,
        valid_14=valid_14,
        valid_23=valid_23,
        valid_24=valid_24,
        valid_34=valid_34,
        valid_123=valid_123,
        valid_124=valid_124,
        valid_134=valid_134,
        valid_234=valid_234,
        valid_all=valid_all,
        onion_reachable=onion_reachable,
        plate_reachable=plate_reachable,
        pot_reachable=pot_reachable,
        goal_reachable=goal_reachable,
        reachable_ok=reachable_ok
    )


def debug_print_layout_result(result):
    jax.debug.print(
        "[layout check] pot={p}, onion={o}, plate={pl}, goal={g}, "
        "count_ok={co}, onion_reachable={or_}, plate_reachable={pr}, "
        "pot_reachable={por}, goal_reachable={gr}, reachable_ok={ro}, valid={v}",
        p=result.pot_count,
        o=result.onion_count,
        pl=result.plate_count,
        g=result.goal_count,
        co=result.count_ok,
        or_=result.onion_reachable,
        pr=result.plate_reachable,
        por=result.pot_reachable,
        gr=result.goal_reachable,
        ro=result.reachable_ok,
        v=result.valid,
    )