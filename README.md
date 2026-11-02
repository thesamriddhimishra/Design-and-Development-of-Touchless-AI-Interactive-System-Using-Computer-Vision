READ through README.md in the project folder for changelogs otherwise run the commands in the setup section of this readme

```bash
python showcase.py
```

This launches a polished multi-mode demo that already includes:

| Mode | What it does | Real-world value |
|------|--------------|------------------|
| **1 Desktop Mouse** | Virtual cursor + tap-pinch click + hold-pinch drag + three-finger right-click | Control any app without touching mouse |
| **2 Presentation** | Point = next, Two fingers = previous, Thumb Up = start, Fist = laser | Control PowerPoint / Google Slides hands-free during a meeting |
| **3 Media** | Play (Thumb Up) / Pause (Open Palm), Point = previous, Fist = next, Pinch+twist = volume knob, Edge-hold = seek | Control music/video while hands are away from keyboard |
| **4 Drawing** | Pinch-to-draw whiteboard | Annotate while presenting |
| **5 Whack-a-Mole** | Fun mini-game | Shows the tracking is fun & responsive |
| **6 AR Workspace** | Pinch a palette icon (Bar/Line/Pie/Live/Files) and drag it out to place a chart on your desktop - Files loads a CSV from `ar_data/`; Fist = dismiss, Two fingers = refresh / cycle Live / reload CSV; layout is saved between launches | Iron-Man-style floating data, not confined to one app window |

Switching between these six modes is done entirely by gesture (hold Open Palm 1s → menu → point + pinch) - see below.

**AI component**: The demo uses MediaPipe’s official pre-trained **Gesture Recognizer** model (a real neural network) for high-level gestures, hybridized with the project’s own rule-based engine for precise pinch and swipe. You will see “AI: Closed_Fist / Open_Palm / …” on screen with a live confidence bar.

### Quick controls inside the demo
- **Thumb Down** → toggle control ON/OFF (safety first); also cancels the mode menu if it's open, and hides/shows the whole AR overlay in AR mode
- **Hold Open Palm for 1s** → open the gesture-driven mode menu (point + pinch to pick a mode) - the main way to switch modes
- **1 / 2 / 3 / 4 / 5 / 6** or **d / p / m / w / g / a** → switch modes directly (backup/dev shortcut)
- **C** → toggle the camera-preview thumbnail (bottom-right)
- **H** → toggle help overlay
- **Q** → quit

## Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
python generate_assets.py   # only needed once, or after deleting assets/
```
