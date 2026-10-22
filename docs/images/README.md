# Images in this folder

`demo_grid.gif`: four 5-second clips made with this node in a real ComfyUI queue (ComfyUI master 1a82e6d0, RTX 5090, Windows 11).
The settings were FCP 1.3B, `umt5_xxl_fp8_e4m3fn_scaled`, seed 42, 50 steps, CFG 3.0, shift 5.0, 768×432, and the official negative prompt.
Each clip is the first and only try for its prompt; nothing was re-rolled or selected. For the GIF they are scaled to 320×180
and every second frame is kept (8 fps; the clips are 16 fps). `demo_grid_first_frame.png` is the first frame of the GIF.

The start images were generated for this demo with MOSS-Image SFT (OpenMOSS, Apache-2.0) in ComfyUI: 1344×768, seed 2026,
50 steps, CFG 3.0. They are not the example images of the Looking Ahead repository.

| position | start image prompt (MOSS-Image) | motion prompt (Looking Ahead) | what happens |
| --- | --- | --- | --- |
| top left | An old wooden windmill with four large sails standing beside a field of red and yellow tulips in the Netherlands, soft clouds, golden hour light, realistic photograph | The four large sails of the windmill turn slowly and steadily in the breeze. The tulips sway gently and the clouds drift slowly to the right. The camera stays still. | The sails turn steadily; their outline changes a little from frame to frame. |
| top right | A vintage steam locomotive pulling green passenger cars across an arched stone bridge in a mountain valley, white smoke rising from the chimney, autumn trees with orange leaves, realistic photograph | The steam locomotive moves steadily from left to right across the stone bridge, pulling its green passenger cars, while white smoke billows from the chimney and drifts back over the train. The camera stays still. | The train crosses the bridge to the right and leaves the frame, smoke trailing behind. |
| bottom left | A golden retriever running along the wet sand of a beach at sunset, small waves rolling in behind it, warm orange sky, low camera angle, realistic photograph | The golden retriever runs to the right along the wet sand, its ears bouncing and its paws kicking up sand, as small waves roll onto the beach behind it. The camera stays still. | The dog runs to the right; the toy in its mouth disappears, and at the end it turns toward the camera. |
| bottom right | A red diamond-shaped kite with a long yellow ribbon tail flying high above a green grassy hillside, blue sky with scattered white clouds, a person holding the string at the bottom left, wide shot, natural daylight, realistic photograph | The red kite rises and dips in the strong wind while its long yellow ribbon tail flutters and curls behind it. The person holding the string takes a small step back. White clouds drift slowly across the blue sky. The camera stays still. | The prompt is not followed: the kite drops and leaves the frame at the bottom left. |

`demo_grid_5b.gif`: the same four start images and motion prompts with the 5B model, in a real ComfyUI queue (ComfyUI master
1a82e6d0, RTX 5090, Windows 11). The settings were FCP 5B, `umt5_xxl_fp8_e4m3fn_scaled`, `wan2.2_vae`, seed 42, 50 steps,
CFG 3.0, shift 5.0, 1024×576, 24 fps, and the official negative prompt. Again each clip is the first and only try. For the GIF
they are scaled to 320×180 and every third frame is kept (8 fps). `demo_grid_5b_first_frame.png` is its first frame.

| position | what happens (5B) |
| --- | --- |
| top left | The sails turn, but their lattice smears and doubles in several frames. |
| top right | The train moves right across the bridge and the locomotive leaves the frame; the smoke merges into the clouds. |
| bottom left | The dog runs to the right along the water. As with 1.3B, the toy in its mouth disappears. |
| bottom right | The kite stays in the air, dipping and swinging while its tail flutters; its shape bends from frame to frame. |

`demo_continuation.gif`: video continuation of the 5B train clip above (`demo_grid_5b.gif`, top right) with the shipped
workflows, unchanged: `workflows/api/looking_ahead_5b_video_continuation_api.json` (left) and
`looking_ahead_video_continuation_api.json` (1.3B, right). Settings: seed 42, 10 s in total, automatic context, prompt "The
steam locomotive keeps moving to the right across the stone bridge, pulling its green passenger cars, while white smoke drifts
back over the train. The camera stays still." Each is the first and only try.
- **Context.** 5B used the last 4.5 s of the clip as context (109 frames at 24 fps) and generated 5.5 s. 1.3B used its last 4.6 s
  (73 frames at 16 fps) and generated 5.4 s.
- **GIF.** It starts at 3.5 s, 320×180 per side, 8 fps. The label on each side says whether the frame is input (the decoded
  context) or generated.
- **What happens.** Both keep the train moving right across the bridge after the input ends. With 5B, the later cars pick up
  red details. With 1.3B, the cars blur and turn brown in the last second.
