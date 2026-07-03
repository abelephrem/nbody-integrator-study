"""Numerical integrators — Euler, RK4, Leapfrog."""

import numpy as np
from bodies import SystemState


def euler_step(state, forces_func, dt):
    positions = state.positions
    velocities = state.velocities

    a = forces_func(state)

    new_pos = positions + dt * velocities
    new_vel = velocities + dt * a

    return SystemState(state.masses, new_pos, new_vel)


def leapfrog_step(state, forces_func, dt):
    positions = state.positions
    velocities = state.velocities
    
    a1 = forces_func(state)  # acceleration at current position
    
    v_half = velocities + 0.5 * dt * a1  # half kick: nudge velocity using a1
    new_pos = positions + dt * v_half  # full drift: move using the half-updated velocity

    # now recompute acceleration at the new positions
    moved_state =  SystemState(state.masses, new_pos, v_half)
    a2 = forces_func(moved_state)  # acceleration after moving
    new_vel = v_half + 0.5 * dt * a2  # half kick: finish the velocity using a2

    return SystemState(state.masses, new_pos, new_vel)


def rk4_step(state, forces_func, dt):
    x = state.positions
    v = state.velocities

    # stage 1: sample the slope right where w eare
    kx1 = v
    kv1 = forces_func(state)  # acceleration at x

    # stage 2: sample at a half-step forward, using stage 1 slopes
    state_k2 = SystemState(state.masses, x + 0.5*dt*kx1, v + 0.5*dt*kv1)
    kx2 = v + 0.5*dt*kv1
    kv2 = forces_func(state_k2)  # acceleration at the nudged position

    # stage 3: another half-step, but using stage 2 slopes
    state_k3 = SystemState(state.masses, x + 0.5*dt*kx2, v + 0.5*dt*kv2)
    kx3 = v + 0.5*dt*kv2
    kv3 = forces_func(state_k3)  # acceleration at the stage 2 nudged position

    # stage 4: a full step forward, using stage 3 slopes
    state_k4 = SystemState(state.masses, x + dt*kx3, v + dt*kv3)
    kx4 = v + dt*kv3
    kv4 = forces_func(state_k4)  # acceleration at the full step position

    # combine: weighted average, middle stages count double
    new_pos = x + (dt/6) * (kx1 + 2*kx2 + 2*kx3 + kx4)
    new_vel = v + (dt/6) * (kv1 + 2*kv2 + 2*kv3 + kv4)

    return SystemState(state.masses, new_pos, new_vel)

