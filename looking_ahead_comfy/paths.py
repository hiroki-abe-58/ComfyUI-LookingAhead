"""Model folder: ComfyUI/models/looking_ahead (extra_model_paths.yaml may add more under the key `looking_ahead`).

The released checkpoints keep the official layout, e.g. models/looking_ahead/fcp_wan21_t2v_1.3b/model_ema.pt."""
import os

import folder_paths

FOLDER = "looking_ahead"
EXTENSIONS = {".pt", ".pth"}


def register():
    path = os.path.join(folder_paths.models_dir, FOLDER)
    if FOLDER in folder_paths.folder_names_and_paths:
        paths, exts = folder_paths.folder_names_and_paths[FOLDER]
        folder_paths.folder_names_and_paths[FOLDER] = (paths if path in paths else [path, *paths], set(exts) | EXTENSIONS)
    else:
        folder_paths.folder_names_and_paths[FOLDER] = ([path], set(EXTENSIONS))


def checkpoints():
    # "fcp_wan21_t2v_1.3b/model_ema.pt" on every OS, so a workflow saved on Windows also loads on Linux and back
    return [name.replace(os.sep, "/") for name in folder_paths.get_filename_list(FOLDER)]


def resolve(name):
    path = folder_paths.get_full_path(FOLDER, name)
    if path is None:
        raise FileNotFoundError(f"{name!r} is not in models/{FOLDER}. Download fcp_wan21_t2v_1.3b/model_ema.pt from "
                                "huggingface.co/TencentARC/Looking-Ahead into ComfyUI/models/looking_ahead/ (see the README).")
    return path
