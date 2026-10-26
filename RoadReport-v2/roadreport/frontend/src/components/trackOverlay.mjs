// Interpolate observed intervals; smoothly track potholes across video playback.
export function boxAtTime(track, time, maxGap = 1.2) {
  if (!Array.isArray(track) || !track.length || !Number.isFinite(time)) return null;
  let left = null;
  for (const point of track) {
    if (point.t >= time) {
      if (Math.abs(point.t - time) < 0.001) return point.box;
      if (!left) return point.t - time <= 0.35 ? point.box : null;
      const gap = point.t - left.t;
      if (gap > maxGap) return null;
      const fraction = gap > 0 ? (time - left.t) / gap : 0;
      return left.box.map((v, i) => v + (point.box[i] - v) * fraction);
    }
    left = point;
  }
  return time - left.t <= 0.35 ? left.box : null;
}
