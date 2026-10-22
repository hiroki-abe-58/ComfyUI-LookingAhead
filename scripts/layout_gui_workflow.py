"""Give the exported example workflow a readable layout (the frontend's API-JSON import stacks nodes on top of each other),
then render it in a running frontend to check. Maintainer tool, like export_gui_workflows.py.

    python scripts/layout_gui_workflow.py workflows/gui/looking_ahead_image_to_video.json --url http://127.0.0.1:PORT --shot out.png [--channel chrome]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# node id -> (x, y, width, height); columns: inputs and loaders | prompts | sampler | output
LAYOUT = {
    "1": (40, 40, 330, 400),     # LoadImage
    "2": (40, 480, 330, 110),    # CLIPLoader
    "5": (40, 630, 330, 60),     # VAELoader
    "6": (40, 730, 330, 60),     # LookingAheadModelLoader
    "3": (410, 40, 420, 230),    # prompt
    "4": (410, 310, 420, 230),   # negative prompt
    "7": (870, 40, 330, 400),    # LookingAheadImageToVideo
    "8": (1240, 40, 380, 460),   # SaveVideo
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow", type=Path)
    ap.add_argument("--url", default=None)
    ap.add_argument("--shot", type=Path, default=None)
    ap.add_argument("--channel", default=None)
    a = ap.parse_args()
    wf = json.loads(a.workflow.read_text(encoding="utf-8"))
    for node in wf["nodes"]:
        x, y, w, h = LAYOUT[str(node["id"])]
        node["pos"] = [x, y]
        node["size"] = [w, h]
    wf.setdefault("extra", {})["ds"] = {"scale": 0.9, "offset": [20, 70]}
    a.workflow.write_text(json.dumps(wf, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if a.url:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(channel=a.channel) if a.channel else p.chromium.launch()
            page = browser.new_page(viewport={"width": 1680, "height": 1000}, locale="en-US")
            page.goto(a.url)
            page.wait_for_function("() => window.app && window.app.graph", timeout=120000)
            page.wait_for_timeout(2500)
            page.keyboard.press("Escape")
            page.evaluate("async (wf) => { await window.app.loadGraphData(wf); }", wf)
            page.wait_for_timeout(1500)
            page.screenshot(path=str(a.shot))
            browser.close()


if __name__ == "__main__":
    main()
