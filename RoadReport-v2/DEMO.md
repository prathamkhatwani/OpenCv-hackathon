# Demo simulation mode

This update adds an explicit **Demo** navigation tab and **Try the sample simulation** button. Demo playback is independent of backend processing. It never substitutes results after a failed upload and never saves simulated findings as real jobs.

## Windows: easiest start

Double-click **START-DEMO.bat** in the RoadReport-v2 folder. It installs frontend dependencies if needed, starts Vite on port 5173 and opens the simulation in your browser. Install Node.js 24 first. Keep the terminal open.

Or run from RoadReport-v2:

```powershell
cd roadreport/frontend
npm ci
npm run demo
```

Open http://127.0.0.1:5173/?demo=1 . For ordinary inspection, start the backend separately using README.md. The demo itself needs no backend, API key or camera permissions.

## Included scenes

| Time | Scene | Illustrated outcome |
|---|---|---|
| 0–6 s | Pothole overlapping the left tyre path; pulse at 5 s | Impact-correlated pothole |
| 6–12 s | Pothole between both tyres; no pulse | Retained pothole; no contact |
| 12–18 s | Shadow with an unrelated shake at 15.8 s | Shake alone does not establish a pothole |

Use Play/Pause, Restart, the three case cards, the seek slider, or the waveform to navigate. Overlays can be toggled off. The evidence panel and reasoning steps follow video time. Download links provide the MP4 and scripted manifest.

The 960×540, 24 fps H.264 video is an original procedural illustration. Its persistent synthetic watermark survives downloading. Boxes, tyre corridors and motion plots are generated from the animation script, not an ML model or physical sensors. This demonstrates the product concept; it is not footage for accuracy evaluation. No paid/stock/reference media is used.

## Reproduce the assets

From RoadReport-v2, in the backend Python environment with ffmpeg installed:

```bash
python roadreport/backend/tools/generate_demo.py
```

This creates `roadreport/frontend/public/demo/road-simulation.mp4` and `manifest.json`. Assets are already bundled; generation is optional. Real uploaded-video analysis is unchanged.

## Verification

Frontend build and source lint passed. Video metadata verified: 18.0 seconds, 432 frames, H.264/yuv420p, 960×540. The Vite server and MP4 range serving were checked locally. Cloud-browser playback could not be verified because it cannot connect to the runtime localhost; test playback in your local browser before presenting.
