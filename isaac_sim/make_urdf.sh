#!/bin/bash
# Expand traquad.xacro and rewrite the mesh paths for the host, so Isaac Sim (running on the host) can find the meshes.
# Uses the ROS container if it is running, otherwise a local xacro (pip install xacro, e.g. in env_isaaclab3) with an
# ament index of the repository packages.
# usage: ./make_urdf.sh <out.urdf> [track_xacro]
#   XACRO_EXTRA: more xacro args
#   track_xacro: optional alternative track.xacro (e.g. an older wheel model); the repo file is restored after.
set -e
OUT=$(readlink -f "$1")
REPO=$(cd "$(dirname "$0")/.." && pwd)
TRACK=$REPO/src/mulinex_description/urdf/track.xacro
CONTAINER=${CONTAINER:-traquad_ros2_humble}
WS=/home/ros/docker_simulation_ws
# Isaac Sim asset values of the track xacro args (the Gazebo defaults of traquad.xacro differ): friction 0.6 measured
# on the lab rubber mat, ankle end stops +-30 deg around the flat-track pose at HFE +-1.47, ankle damping 0.01,
# bounding-box track body colliders. XACRO_EXTRA comes after and overrides them.
XACRO_ARGS="wheel_mu:=0.6 ankle_center:=0.1008 ankle_range:=0.5236 ankle_damping:=0.01 body_box:=true $XACRO_EXTRA"
if [ -n "$2" ]; then cp "$TRACK" /tmp/track.xacro.bak; cp "$2" "$TRACK"; fi
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
  docker exec "$CONTAINER" bash -c "source /opt/ros/humble/setup.bash && cd $WS && source install/setup.bash && \
    xacro install/mulinex_description/share/mulinex_description/urdf/traquad.xacro $XACRO_ARGS" > "$OUT"
  MESHES="file://$WS/install/mulinex_description/share/mulinex_description/meshes//"
else
  PREFIX=$(mktemp -d)
  mkdir -p "$PREFIX/share/ament_index/resource_index/packages"
  for p in "$REPO"/src/*/; do
    n=$(basename "$p"); touch "$PREFIX/share/ament_index/resource_index/packages/$n"; ln -s "$REPO/src/$n" "$PREFIX/share/$n"
  done
  AMENT_PREFIX_PATH="$PREFIX" xacro "$REPO/src/mulinex_description/urdf/traquad.xacro" $XACRO_ARGS > "$OUT"
  rm -rf "$PREFIX"
  MESHES="file://$PREFIX/share/mulinex_description/meshes//"
fi
if [ -n "$2" ]; then cp /tmp/track.xacro.bak "$TRACK"; fi
sed -i "s|$MESHES|$REPO/src/mulinex_description/meshes/|g" "$OUT"
echo "written $OUT ($XACRO_ARGS)"
