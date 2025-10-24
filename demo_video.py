#!/usr/bin/env python3
"""
Converted from notebooks/video_predictor_example.ipynb -> demo/demo_video.py
A minimal runnable script that:
 - builds the SAM2 video predictor
 - initializes inference state for a directory of JPEG frames
 - performs a sample interaction (one positive click)
 - propagates the prediction through the video
 - saves visualization images to an output directory

Usage example:
 python demo_video.py --video_dir ./videos/bedroom --checkpoint ../checkpoints/sam2.1_hiera_large.pt --config configs/sam2.1/sam2.1_hiera_l.yaml --output_dir ./out --device cuda

Note: this script assumes `sam2` package is installed and checkpoints/configs exist at provided paths.
"""

import argparse
import os
import numpy as np
import torch
import matplotlib
import os
# choose interactive backend only if a display is available; otherwise use Agg
if os.environ.get('DISPLAY'):
    try:
        matplotlib.use('TkAgg')   # requires python3-tk
    except Exception:
        matplotlib.use('Agg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
from sam2.utils.misc import AsyncVideoFrameLoader
from collections import deque


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--video_dir', type=str, required=True, help='Directory with JPEG frames named as integers (e.g. 00000.jpg)')
    p.add_argument('--end_frame', type=int, default=None, help='Optionally only process up to this frame index (exclusive)')
    p.add_argument('--checkpoint', type=str, required=True, help='Path to sam2 checkpoint (.pt)')
    p.add_argument('--config', type=str, required=True, help='Path to sam2 config (.yaml)')
    p.add_argument('--device', type=str, default=None, help='device: cuda, cpu, or leave empty to auto-detect')
    p.add_argument('--output_dir', type=str, default='./out', help='Where to save visualizations')
    p.add_argument('--frame_stride', type=int, default=1, help='How often to save frames to output')
    p.add_argument('--chunk_size', type=int, default=0, help='Process video in chunks of this many frames (0=disable)')
    p.add_argument('--overlap', type=int, default=1, help='Overlap frames between chunks to seed next chunk (recommended 1)')
    p.add_argument('--offload_video_to_cpu', action='store_true', help='Keep frames on CPU to reduce GPU memory')
    return p.parse_args()


def select_device(preferred=None):
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_predictor(cfg_path, checkpoint_path, device):
    # import here to allow the script to start even when sam2 isn't installed
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(cfg_path, checkpoint_path, device=device)
    return predictor


def load_frame_names(video_dir):
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    # sort by integer filename (expected '<idx>.jpg' or '00001.jpg')
    try:
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    except Exception:
        frame_names.sort()
    # if end_frame is not None:
    #     frame_names = frame_names[:end_frame]
    return frame_names


def overlay_and_save(image_path, masks_dict, output_path):
    img = Image.open(image_path).convert('RGB')
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(img)
    for obj_id, mask in masks_dict.items():
        show_mask(mask, ax, obj_id=obj_id)
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def build_inference_state_from_paths(predictor, img_paths, offload_video_to_cpu=False):
    """Build an inference_state like predictor.init_state but from an explicit list of image paths.
    Uses AsyncVideoFrameLoader to avoid loading all frames at once.
    """
    # mean/std tensors as in utils.misc
    img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
    img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
    images = AsyncVideoFrameLoader(
        img_paths=img_paths,
        image_size=predictor.image_size,
        offload_video_to_cpu=offload_video_to_cpu,
        img_mean=img_mean,
        img_std=img_std,
        compute_device=predictor.device,
    )
    inference_state = {}
    inference_state["images"] = images
    inference_state["num_frames"] = len(images)
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    inference_state["offload_state_to_cpu"] = False
    inference_state["video_height"] = images.video_height
    inference_state["video_width"] = images.video_width
    inference_state["device"] = predictor.device
    inference_state["storage_device"] = predictor.device
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["constants"] = {}
    inference_state["obj_id_to_idx"] = {}
    inference_state["obj_idx_to_id"] = {}
    inference_state["obj_ids"] = []
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["frames_tracked_per_obj"] = {}
    # warm up backbone on frame 0
    predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
    return inference_state


def main():
    args = parse_args()
    device = select_device(args.device)
    print(f'Using device: {device}')

    predictor = build_predictor(args.config, args.checkpoint, device=device)

    video_dir = args.video_dir
    frame_names = load_frame_names(video_dir)
    if len(frame_names) == 0:
        raise SystemExit(f'No JPEG frames found in {video_dir}')

    os.makedirs(args.output_dir, exist_ok=True)

    # chunking strategy: if chunk_size <= 0, we'll process whole video at once; in all cases,
    # defer any heavy loading/init until AFTER the GUI window is closed.
    use_chunking = args.chunk_size and args.chunk_size > 0
    inference_state = None

    # sample interaction: positive click on frame 0
    ann_frame_idx = 0
    ann_obj_id = 1

    # ...existing code...
    # Show the interacted frame and let the user click a point with the mouse (returns (x,y) in image coords).
    image_path = os.path.join(video_dir, frame_names[ann_frame_idx])
    img = Image.open(image_path).convert('RGB')

    # Try to switch to an interactive backend if possible (may fail in headless env).
    try:
        plt.switch_backend('tkagg')
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(10, 6))
    base_im = ax.imshow(img)
    # layout: make main image narrower so we have a right-side panel for coords/buttons
    ax.set_position([0.03, 0.05, 0.70, 0.9])
    ax.set_title('Click to select positive points (masks update live; close window when done)')
    coords = []

    # state for incremental visualization
    mask_artists = []
    point_artists = []
    interacted_masks = {}
    did_interactive_infer = False
    interacted_out_obj_ids = None
    interacted_out_mask_logits = None
    # lightweight state for live preview on the first frame only
    gui_preview_state = None

    # Instances: each instance has unique obj_id and a list of points
    instances = []
    next_instance_id = 1
    current_instance_idx = 0
    instance_outputs = {}

    def make_instance():
        nonlocal next_instance_id
        inst = {'obj_id': next_instance_id, 'points': [], 'color': plt.get_cmap('tab10')((next_instance_id-1) % 10)}
        next_instance_id += 1
        return inst

    # start with one instance
    instances.append(make_instance())

    # create a coords panel on the right
    ax_coords = fig.add_axes([0.74, 0.25, 0.22, 0.7])
    ax_coords.axis('off')

    def render_points_and_ui():
        # keep base image only: remove any additional images
        for im in ax.get_images()[1:]:
            try:
                im.remove()
            except Exception:
                pass
        # redraw masks first
        redraw_masks()
        # remove previous point artists
        for pa in list(point_artists):
            try:
                pa.remove()
            except Exception:
                pass
        point_artists.clear()
        # draw points for each instance
        for idx, inst in enumerate(instances):
            pts = np.array(inst['points'], dtype=np.float32) if len(inst['points']) > 0 else None
            color = inst.get('color', (0, 1, 0, 1))
            if pts is not None:
                coll = ax.scatter(pts[:, 0], pts[:, 1], color=color, marker='*', s=150, edgecolor='white')
                point_artists.append(coll)
        # update coords panel
        coord_lines = []
        for idx, inst in enumerate(instances):
            header = f"Instance {inst['obj_id']}{' <-' if idx==current_instance_idx else ''}"
            coords_text = '\n'.join([f"{i}: ({p[0]:.1f},{p[1]:.1f})" for i, p in enumerate(inst['points'])])
            coord_lines.append(header)
            coord_lines.append(coords_text if coords_text else '(no points)')
            coord_lines.append('')
        ax_coords.clear()
        ax_coords.text(0, 1, '\n'.join(coord_lines), va='top', fontsize=9, family='monospace')
        ax_coords.axis('off')
        fig.canvas.draw_idle()

    # --- small GUI buttons for instance management (4 buttons) ---
    ax_btn_add = fig.add_axes([0.78, 0.12, 0.18, 0.06])
    ax_btn_del_point = fig.add_axes([0.78, 0.04, 0.18, 0.06])

    btn_add = Button(ax_btn_add, 'Add Instance')
    btn_del_point = Button(ax_btn_del_point, 'Delete Instance')

    def on_add(event):
        nonlocal current_instance_idx
        instances.append(make_instance())
        current_instance_idx = len(instances) - 1
        # keep numbering contiguous
        renumber_instances()
        render_points_and_ui()

    def on_del_point(event):
        nonlocal current_instance_idx
        # delete the entire current instance. If it's the only instance, clear its points instead.
        if len(instances) <= 1:
            inst = instances[0]
            inst['points'].clear()
            obj_id = inst.get('obj_id')
            if obj_id in instance_outputs:
                del instance_outputs[obj_id]
            render_points_and_ui()
            return

        inst = instances.pop(current_instance_idx)
        obj_id = inst.get('obj_id')
        if obj_id in instance_outputs:
            del instance_outputs[obj_id]
        # renumber remaining instances so their counts are contiguous
        renumber_instances()
        # adjust current index
        if current_instance_idx >= len(instances):
            current_instance_idx = len(instances) - 1
        render_points_and_ui()

    def renumber_instances():
        # reassign obj_id sequentially starting from 1 and update colors
        nonlocal next_instance_id
        old_to_new = {}
        for idx, inst in enumerate(instances):
            old = inst.get('obj_id')
            new = idx + 1
            if old != new:
                old_to_new[old] = new
                inst['obj_id'] = new
            # update color according to new id
            inst['color'] = plt.get_cmap('tab10')((inst['obj_id']-1) % 10)
        # remap instance_outputs keys from old->new where possible
        if old_to_new:
            new_outputs = {}
            for key, mask in list(instance_outputs.items()):
                if key in old_to_new:
                    new_outputs[old_to_new[key]] = mask
                else:
                    new_outputs[key] = mask
            instance_outputs.clear()
            instance_outputs.update(new_outputs)
        next_instance_id = len(instances) + 1

    btn_add.on_clicked(on_add)
    btn_del_point.on_clicked(on_del_point)

    # --- Reset All button (restore initial state) ---
    ax_btn_reset = fig.add_axes([0.78, -0.02, 0.18, 0.06])
    btn_reset = Button(ax_btn_reset, 'Reset All')

    def on_reset(event):
        nonlocal next_instance_id, current_instance_idx, did_interactive_infer
        nonlocal interacted_out_obj_ids, interacted_out_mask_logits
        # reset ids and instances
        next_instance_id = 1
        instances.clear()
        instances.append(make_instance())
        current_instance_idx = 0
        # clear outputs and flags
        instance_outputs.clear()
        coords.clear()
        interacted_masks.clear()
        did_interactive_infer = False
        interacted_out_obj_ids = None
        interacted_out_mask_logits = None
        # remove any artists
        for a in list(mask_artists):
            try:
                a.remove()
            except Exception:
                pass
        mask_artists.clear()
        try:
            for p in list(point_artists):
                try:
                    p.remove()
                except Exception:
                    pass
        except NameError:
            pass
        if 'point_artists' in globals() or 'point_artists' in locals():
            point_artists.clear()
        render_points_and_ui()

    btn_reset.on_clicked(on_reset)

    def redraw_masks(masks_dict=None):
        # remove previous mask artists
        for art in list(mask_artists):
            try:
                art.remove()
            except Exception:
                pass
        mask_artists.clear()
        # draw new masks (from provided masks_dict or from instance_outputs)
        src = masks_dict if masks_dict is not None else instance_outputs
        for obj_id, mask in src.items():
            try:
                mask_arr = np.asarray(mask)
                if mask_arr.ndim > 2:
                    mask_arr = np.squeeze(mask_arr)
                if mask_arr.ndim > 2:
                    mask_arr = mask_arr.reshape(mask_arr.shape[-2], mask_arr.shape[-1])
                h, w = mask_arr.shape[-2], mask_arr.shape[-1]
                mask2d = mask_arr.reshape(h, w)
            except Exception:
                # skip malformed masks
                continue
            # color: use instance color if available, else orange
            color = np.array([1.0, 0.55, 0.0, 0.45], dtype=np.float32)
            for inst in instances:
                if inst['obj_id'] == obj_id:
                    c = inst.get('color', None)
                    if c is not None:
                        color = np.array([c[0], c[1], c[2], 0.45], dtype=np.float32)
                    break
            mask_img = np.zeros((h, w, 4), dtype=np.float32)
            mask_img[..., :3] = color[:3]
            mask_img[..., 3] = mask2d.astype(np.float32) * color[3]
            art = ax.imshow(mask_img, origin='upper')
            mask_artists.append(art)
        fig.canvas.draw_idle()

    def onclick(event):
        nonlocal interacted_masks, did_interactive_infer, current_instance_idx
        nonlocal interacted_out_obj_ids, interacted_out_mask_logits, gui_preview_state
        # only accept clicks on the main image axes
        if event.inaxes is None or event.inaxes is not ax:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        # ensure click is within image pixel bounds
        try:
            W, H = img.size
            if x < 0 or x > W or y < 0 or y > H:
                return
        except Exception:
            pass
        # add to current instance
        inst = instances[current_instance_idx]
        inst['points'].append((x, y))
        coords.append((x, y))
        # Live preview: build a tiny state for the first frame only (if not yet), then run inference
        try:
            if gui_preview_state is None:
                first_img_path = os.path.join(video_dir, frame_names[ann_frame_idx])
                gui_preview_state = build_inference_state_from_paths(
                    predictor, [first_img_path], offload_video_to_cpu=args.offload_video_to_cpu
                )
            points_arr = np.array(inst['points'], dtype=np.float32)
            labels = np.ones(len(inst['points']), np.int32)
            if len(points_arr) > 0:
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=gui_preview_state,
                    frame_idx=0,  # local index within the preview state
                    obj_id=inst['obj_id'],
                    points=points_arr,
                    labels=labels,
                )
                # Update instance_outputs for on-canvas rendering
                for i, out_obj_id in enumerate(out_obj_ids):
                    instance_outputs[out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()
                interacted_out_obj_ids = out_obj_ids
                interacted_out_mask_logits = out_mask_logits
                did_interactive_infer = True
        except Exception as e:
            print('Live preview inference failed:', e)
        # redraw UI with updated masks/points
        render_points_and_ui()
    
    # register the onclick handler and show the interactive window (outside the handler)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    print('Please click positive points on the displayed image window (close window when done)...')
    if matplotlib.get_backend().lower().startswith('agg'):
        # no interactive backend available
        raise SystemExit('No interactive matplotlib backend available (Agg selected). '
                         'Run this script on a machine with DISPLAY or enable X forwarding, '
                         'and install a GUI backend (e.g. python3-tk).')
    try:
        plt.show()
    except Exception:
        # fallback: keep event loop alive until window closed
        while plt.fignum_exists(fig.number):
            plt.pause(0.1)

    # after window closed, ensure we have at least one point
    if len(coords) == 0:
        raise SystemExit('No point selected. Exiting.')

    # only create/init inference state AFTER GUI window is closed
    print('Preparing model state and adding initial point(s)...')
    interacted_masks = {}
    if inference_state is None:
        if use_chunking:
            # build only the first chunk state; the rest will be built during propagation
            start = 0
            end = min(len(frame_names), args.chunk_size)
            chunk_paths = [os.path.join(video_dir, fn) for fn in frame_names[start:end]]
            inference_state = build_inference_state_from_paths(
                predictor, chunk_paths, offload_video_to_cpu=args.offload_video_to_cpu
            )
        else:
            # initialize inference state for the whole video now
            print('Initializing inference state (loading frames/features as needed)...')
            inference_state = predictor.init_state(
                video_path=video_dir,
                offload_video_to_cpu=args.offload_video_to_cpu,
                async_loading_frames=True,
            )
            predictor.reset_state(inference_state)

    # Add points for each instance now
    for inst in instances:
        if len(inst['points']) == 0:
            continue
        pts = np.array(inst['points'], dtype=np.float32)
        labels = np.ones(len(pts), np.int32)
        try:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=inst['obj_id'],
                points=pts,
                labels=labels,
            )
            for i, out_obj_id in enumerate(out_obj_ids):
                interacted_masks[out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()
        except Exception as e:
            print(f'Warning: add_new_points_or_box failed for instance {inst.get("obj_id")}: {e}')

    # Free the tiny GUI preview state to reclaim GPU memory before heavy processing
    try:
        if 'gui_preview_state' in locals() and gui_preview_state is not None:
            del gui_preview_state
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    except Exception:
        pass


    # save the interacted frame visualization
    interaction_out = os.path.join(args.output_dir, f'interacted_frame_{ann_frame_idx}.png')
    overlay_and_save(os.path.join(video_dir, frame_names[ann_frame_idx]), interacted_masks, interaction_out)
    print(f'Saved interacted frame visualization to: {interaction_out}')

    # propagate: either whole video at once, or in chunks with mask handoff
    bboxs_allframes = []
    if not use_chunking:
        print('Propagating predictions through the video (this may take time)...')
        for out_frame_idx, obj_ids_prop, video_res_masks in predictor.propagate_in_video(inference_state):
            # compute bboxes only on required stride
            if out_frame_idx % args.frame_stride != 0:
                continue
            for i, oid in enumerate(obj_ids_prop):
                arr = (video_res_masks[i] > 0).squeeze().detach().cpu().numpy()
                if arr.ndim > 2:
                    arr = np.squeeze(arr)
                if arr.ndim > 2:
                    arr = arr.reshape(arr.shape[-2], arr.shape[-1])
                mask_bool = arr > 0
                ys, xs = np.where(mask_bool)
                if ys.size == 0:
                    continue
                y0, y1 = int(ys.min()), int(ys.max())
                x0, x1 = int(xs.min()), int(xs.max())
                delta = 10
                x0 = max(0, x0 - delta)
                y0 = max(0, y0 - delta)
                x1 = min(arr.shape[1]-1, x1 + delta)
                y1 = min(arr.shape[0]-1, y1 + delta)
                bboxs_allframes.append((out_frame_idx, oid, (x0, y0, x1, y1)))
        print('Propagation done.')
    else:
        print(f'Propagating in chunks of {args.chunk_size} with {args.overlap} frame overlap...')
        total = len(frame_names)
        prev_overlap_masks = None  # list[dict[obj_id -> mask]] for last `overlap` frames of previous chunk
        start = 0
        while start < total:
            # compute chunk range; add overlap on the left except for the first chunk
            if start == 0:
                prepend = 0
                chunk_start = 0
                chunk_end = min(total, start + args.chunk_size)  # exclusive
                chunk_paths = [os.path.join(video_dir, fn) for fn in frame_names[chunk_start:chunk_end]]
                # Reuse the already-initialized inference_state for the first chunk to avoid duplicate GPU usage
                chunk_state = inference_state
            else:
                prepend = min(args.overlap, start)
                chunk_start = start - prepend
                chunk_end = min(total, start + args.chunk_size)
                chunk_paths = [os.path.join(video_dir, fn) for fn in frame_names[chunk_start:chunk_end]]
                # build state for this chunk only
                chunk_state = build_inference_state_from_paths(
                    predictor, chunk_paths, offload_video_to_cpu=args.offload_video_to_cpu
                )
                # seed with masks for the overlapped first `prepend` frames (indices 0..prepend-1)
                if prev_overlap_masks is not None and prepend > 0:
                    try:
                        seed_frames = prev_overlap_masks[-prepend:]
                    except Exception:
                        seed_frames = []
                    for f_idx, frame_masks in enumerate(seed_frames):
                        for oid, m in frame_masks.items():
                            try:
                                predictor.add_new_mask(
                                    inference_state=chunk_state,
                                    frame_idx=f_idx,
                                    obj_id=oid,
                                    mask=m.astype(np.uint8),
                                )
                            except Exception as e:
                                print(f'Warning: add_new_mask failed for obj {oid} on new chunk frame {f_idx}: {e}')

            # determine propagation range within this chunk (skip overlapped frame if any)
            if start == 0:
                local_start_idx = 0
            else:
                local_start_idx = prepend  # e.g., 1 when overlap=1
            max_frames = len(chunk_paths) - local_start_idx

            # per-chunk buffer to keep only the last `args.overlap` frames' masks
            overlap_buffer = deque(maxlen=args.overlap)

            for out_frame_idx_local, obj_ids_prop, video_res_masks in predictor.propagate_in_video(
                chunk_state, start_frame_idx=local_start_idx, max_frame_num_to_track=max_frames
            ):
                # map local index to global frame index
                global_idx = chunk_start + out_frame_idx_local
                stride_ok = (global_idx % args.frame_stride == 0)
                # collect masks for this frame (for both output saving and overlap seeding)
                frame_masks_dict = {}
                for i, oid in enumerate(obj_ids_prop):
                    arr = (video_res_masks[i] > 0).squeeze().detach().cpu().numpy()
                    if arr.ndim > 2:
                        arr = np.squeeze(arr)
                    if arr.ndim > 2:
                        arr = arr.reshape(arr.shape[-2], arr.shape[-1])
                    mask_bool = arr > 0
                    ys, xs = np.where(mask_bool)
                    if ys.size != 0 and stride_ok:
                        y0, y1 = int(ys.min()), int(ys.max())
                        x0, x1 = int(xs.min()), int(xs.max())
                        delta = 10
                        x0 = max(0, x0 - delta)
                        y0 = max(0, y0 - delta)
                        x1 = min(arr.shape[1]-1, x1 + delta)
                        y1 = min(arr.shape[0]-1, y1 + delta)
                        bboxs_allframes.append((global_idx, oid, (x0, y0, x1, y1)))
                    # save visualization for this frame 调用overlay_and_save
                    mask_np = (video_res_masks[i] > 0).squeeze().detach().cpu().numpy().astype(np.uint8)
                    frame_masks_dict[oid] = mask_np
                # append to overlap buffer (keeps only last `args.overlap` frames)
                overlap_buffer.append(frame_masks_dict)
                # optionally save visualization for this frame
                if stride_ok and len(frame_masks_dict) > 0 and out_frame_idx_local - args.overlap == 0:
                    out_viz_path = os.path.join(args.output_dir, f'frame_{global_idx:05d}.png')
                    overlay_and_save(
                        os.path.join(video_dir, frame_names[global_idx]),
                        frame_masks_dict,
                        out_viz_path
                    )
                if global_idx == args.end_frame:
                    break
            # keep last `args.overlap` frames' masks for seeding next chunk
            prev_overlap_masks = list(overlap_buffer)
                
                
            # advance to next chunk start
            start += args.chunk_size

    # save all bboxs to an npz file
    bboxs_outpath = os.path.join(args.output_dir, 'bboxes_0-{}.npz'.format(args.end_frame if args.end_frame is not None else total))
    # convert to structured array for saving
    dtype = np.dtype([('frame_idx', np.int32), ('obj_id', np.int32), ('bbox', np.int32, (4,))])
    bboxs_array = np.array(bboxs_allframes, dtype=dtype)
    np.savez_compressed(bboxs_outpath, bboxs=bboxs_array)
    print(f'Saved all bounding boxes to: {bboxs_outpath}')
    print('Done. Visualizations saved to:', args.output_dir)


if __name__ == '__main__':
    main()
