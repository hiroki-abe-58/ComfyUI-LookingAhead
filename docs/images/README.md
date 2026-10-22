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
