# animate_stl_sequence.py
"""
Animate a sequence of STL meshes (one STL per time frame) using PyVista.

Usage:
    python animate_stl_sequence.py --label_dir "C:/.../patient002/LV" \
        --out "C:/.../LV_animation.mp4" --framerate 10 --offscreen

Requirements:
    pip install pyvista natsort imageio
"""
import argparse
from pathlib import Path
import sys
import platform
import imageio
import pyvista as pv
import natsort

def load_sorted_stls(folder: Path, pattern: str = "*.stl"):
    files = list(folder.glob(pattern)) + list(folder.glob(pattern.upper()))
    if not files:
        raise FileNotFoundError(f"No STL files found in {folder} matching {pattern}")
    files = natsort.natsorted(files)
    return files

def _try_open_movie(plotter: pv.Plotter, out_path: Path, framerate: int):
    """
    Try to open a movie for the plotter. Return True on success, False on failure.
    """
    try:
        plotter.open_movie(str(out_path), framerate=framerate)
        return True
    except Exception:
        return False

def animate_sequence(
    label_dir: Path,
    out_path: Path,
    framerate: int = 10,
    offscreen: bool = True,
    decimate_ratio: float | None = None,
    camera_position: str | None = "xy",
    show_axes: bool = False,
):
    files = load_sorted_stls(label_dir)
    # Try to create a plotter that supports off-screen rendering if requested.
    use_movie = False
    use_screenshots = False

    # On Linux, start_xvfb is available via pyvista helper (but deprecated).
    # We avoid calling start_xvfb on Windows.
    if offscreen and platform.system() == "Linux":
        try:
            pv.start_xvfb()  # best-effort on Linux headless
        except Exception:
            pass

    # Create plotter with off_screen flag (PyVista will try best available backend)
    try:
        plotter = pv.Plotter(off_screen=bool(offscreen), window_size=(1024, 1024))
    except Exception:
        # If creating an off-screen plotter fails, fall back to interactive
        plotter = pv.Plotter(off_screen=False, window_size=(1024, 1024))
        offscreen = False

    # Try to open movie (this may fail on some Windows setups)
    if offscreen:
        if _try_open_movie(plotter, out_path, framerate):
            use_movie = True
        else:
            # fallback: we'll capture screenshots and assemble them with imageio
            use_screenshots = True
            print("[animate] Warning: movie writer not available; falling back to screenshot capture.")
    else:
        # interactive mode: still allow saving frames via screenshot
        use_screenshots = True

    actor = None
    text_id = None
    frames = []  # used only if use_screenshots

    # Initialize the plotter window/context once
    first_mesh = pv.read(str(files[0]))
    actor = plotter.add_mesh(first_mesh, color="lightcoral", smooth_shading=True)
    if show_axes:
        plotter.add_axes()
    if camera_position is not None:
        try:
            plotter.camera_position = camera_position
        except Exception:
            pass
    text_id = plotter.add_text(f"Frame 0", font_size=14)
    # show once to initialize the rendering context (auto_close=False keeps it open)
    plotter.show(auto_close=False)

    for i, f in enumerate(files):
        mesh = pv.read(str(f))
        # optional decimation
        if decimate_ratio is not None and 0.0 < decimate_ratio < 1.0:
            try:
                mesh = mesh.decimate_pro(target_reduction=1.0 - decimate_ratio, preserve_topology=True)
            except Exception:
                pass

        # Update actor geometry
        try:
            actor.mapper.SetInputData(mesh)
        except Exception:
            # fallback: remove and re-add actor
            try:
                plotter.remove_actor(actor)
            except Exception:
                pass
            actor = plotter.add_mesh(mesh, color="lightcoral", smooth_shading=True)

        # Update text
        try:
            plotter.update_text(f"Frame {i:02d}", name=text_id)
        except Exception:
            # re-add text if update fails
            try:
                plotter.remove_actor(text_id)
            except Exception:
                pass
            text_id = plotter.add_text(f"Frame {i:02d}", font_size=14)

        # Render and record
        if use_movie:
            plotter.write_frame()
        elif use_screenshots:
            # render and capture screenshot as numpy array
            img = plotter.screenshot(return_img=True)
            frames.append(img)
        else:
            # interactive only; just render a frame (no saving)
            plotter.render()

    # Close plotter
    plotter.close()

    # If we captured screenshots, write them to a movie file now
    if use_screenshots:
        # Choose writer based on output extension
        out_ext = out_path.suffix.lower()
        if out_ext in (".mp4", ".mov", ".m4v"):
            imageio.mimsave(str(out_path), frames, fps=framerate, macro_block_size=None)
        elif out_ext in (".gif",):
            imageio.mimsave(str(out_path), frames, fps=framerate)
        else:
            # default to mp4 if unknown extension
            fallback = out_path.with_suffix(".mp4")
            imageio.mimsave(str(fallback), frames, fps=framerate, macro_block_size=None)
            print(f"[animate] Saved to fallback path: {fallback}")

    print(f"Saved animation to: {out_path if out_path.exists() else 'output (see messages)'}")


def main():
    parser = argparse.ArgumentParser(description="Animate STL sequence with PyVista")
    parser.add_argument("--label_dir", required=True, help="Folder containing STL sequence (one file per frame)")
    parser.add_argument("--out", required=True, help="Output MP4/GIF path")
    parser.add_argument("--framerate", type=int, default=10)
    parser.add_argument("--offscreen", action="store_true", help="Render off-screen (best-effort)")
    parser.add_argument("--decimate_ratio", type=float, default=None, help="Fraction of faces to keep (0-1), e.g. 0.2")
    parser.add_argument("--camera", default="xy", help="PyVista camera preset or None")
    parser.add_argument("--show_axes", action="store_true")
    args = parser.parse_args()

    label_dir = Path(args.label_dir)
    out_path = Path(args.out)

    animate_sequence(
        label_dir=label_dir,
        out_path=out_path,
        framerate=args.framerate,
        offscreen=args.offscreen,
        decimate_ratio=args.decimate_ratio,
        camera_position=args.camera,
        show_axes=args.show_axes,
    )

if __name__ == "__main__":
    main()
