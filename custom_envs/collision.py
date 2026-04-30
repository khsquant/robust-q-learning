"""Compatibility helpers for older MuJoCo Playground collision utilities."""

import jax.numpy as jp


def geoms_colliding(data, geom1: int, geom2: int):
  """Returns whether two geom ids are present in the MJX contact buffer."""
  contact_geoms = jp.asarray(data.contact.geom).reshape((-1, 2))
  pair_12 = (contact_geoms[:, 0] == geom1) & (contact_geoms[:, 1] == geom2)
  pair_21 = (contact_geoms[:, 0] == geom2) & (contact_geoms[:, 1] == geom1)
  return jp.any(pair_12 | pair_21)
