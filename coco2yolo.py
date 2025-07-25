import json, shutil
from pathlib import Path
#region params
DATASET_ROOT = Path("")
COCO_ROOT    = Path("")
SPLITS       = ["train", "valid", "test"]
COPY_IMAGES  = True   #if false then it only references the images
IM_DIR   = DATASET_ROOT / "images"
LAB_DIR  = DATASET_ROOT / "labels"
LIST_DIR = DATASET_ROOT
YAML_OUT = DATASET_ROOT / "data.yaml"
#endregion params


#region funcs
def norm_poly(poly, w, h):
    return [ (poly[i] / w if i % 2 == 0 else poly[i] / h) for i in range(len(poly)) ]

def coco_to_yolo_bbox(b, w, h):
    x, y, bw, bh = b
    return (x + bw/2)/w, (y + bh/2)/h, bw/w, bh/h

def convert_split(split):
    jp = COCO_ROOT / split / "_annotations.coco.json"
    if not jp.exists():
        print(f"ERR: NO JSON FOUND: {jp}")
        return

    out_lab = LAB_DIR / split
    out_lab.mkdir(parents=True, exist_ok=True)
    list_file = LIST_DIR / f"{'val' if split=='valid' else split}.txt"
    paths = []
    coco = json.loads(jp.read_text(encoding="utf-8"))
    cats = {c["id"]: i for i, c in enumerate(coco["categories"])}
    imgs = {im["id"]: im for im in coco["images"]}
    anns_by_img = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    coco_img_dir = COCO_ROOT / split
    out_img_dir = IM_DIR / split
    out_img_dir.mkdir(parents=True, exist_ok=True)

    for img_id, img in imgs.items():
        w, h = img["width"], img["height"]
        fname = img["file_name"]
        stem  = Path(fname).stem
        src_img = coco_img_dir / fname
        dst_img = out_img_dir / fname

        if COPY_IMAGES:
            if not dst_img.exists():
                if not src_img.exists():
                    print(f"[ERR]: NO IMAGE FOUND: {src_img}")
                    continue
                shutil.copy(src_img, dst_img)
            img_path_for_list = dst_img.as_posix()
        else: img_path_for_list = src_img.as_posix()

        paths.append(img_path_for_list)

        lines = []
        for ann in anns_by_img.get(img_id, []):
            cid = cats[ann["category_id"]]
            if ann.get("segmentation"):
                for seg in ann["segmentation"]:
                    coords = norm_poly(seg, w, h)
                    lines.append(str(cid) + " " + " ".join(f"{v:.6f}" for v in coords))
            elif ann.get("bbox"):
                xc, yc, bw, bh = coco_to_yolo_bbox(ann["bbox"], w, h)
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        (out_lab / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    list_file.write_text("\n".join(sorted(paths)), encoding="utf-8")
    print(f"DONE {split}: {len(paths)} img processed")

def write_yaml():
    if YAML_OUT.exists():
        print("Exisiting data.yaml, no overwrite.")
        return
    coco = None
    for s in SPLITS:
        jp = COCO_ROOT / s / "_annotations.coco.json"
        if jp.exists():
            coco = json.loads(jp.read_text(encoding="utf-8"))
            break
    if coco is None:
        print("ERR: NO JSON FOUND")
        return

    names = {i: c["name"] for i, c in enumerate(coco["categories"])}
    yaml_text = f"""# Auto-generated for YOLOv8 segment
    path: {DATASET_ROOT.as_posix()}

    train: train.txt
    val: val.txt
    test: test.txt

    names:
    """
    for i, n in names.items():
        yaml_text += f"  {i}: {n}\n"

    YAML_OUT.write_text(yaml_text, encoding="utf-8")
    print("data.yaml created successfully.")
#endregion funcs

if __name__ == "__main__":
    for s in SPLITS:
        convert_split(s)
    write_yaml()
