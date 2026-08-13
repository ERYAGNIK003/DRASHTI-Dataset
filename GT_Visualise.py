import os
import cv2
import math
import random
import shutil
import argparse
import numpy as np


W = 3840
H = 2160


def rotate_cart(origin, point, angle):
    ox, oy = origin
    px, py = point

    oy = -oy
    py = -py

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    qy = -qy

    return int(qx), int(qy)


def read_dota_label(label_path):
    """
    DRASHTI-HaOBB label format:

    flightHeight: <height>        [optional first line]

    x1 y1 x2 y2 x3 y3 x4 y4 class difficulty angle
    """

    flight_height = None
    objects = []

    with open(label_path, "r") as f:
        first_valid_line_checked = False

        for idx, line in enumerate(f):
            line = line.strip()

            if not line:
                continue

            if not first_valid_line_checked:
                first_valid_line_checked = True

                if line.startswith("flightHeight"):
                    flight_height = float(line.split(":")[1])
                    continue

            parts = line.split()

            if len(parts) < 11:
                continue

            pts = list(map(float, parts[:8]))

            cls = parts[8]
            difficulty = int(parts[9])
            angle = float(parts[10])

            points = np.array(
                [
                    (pts[0], pts[1]),
                    (pts[2], pts[3]),
                    (pts[4], pts[5]),
                    (pts[6], pts[7]),
                ],
                dtype=np.float32,
            )

            # Clip coordinates to image boundaries
            points[:, 0] = np.clip(points[:, 0], 0, W - 1)
            points[:, 1] = np.clip(points[:, 1], 0, H - 1)

            xmin = int(points[:, 0].min())
            ymin = int(points[:, 1].min())
            xmax = int(points[:, 0].max())
            ymax = int(points[:, 1].max())

            cx = int((xmin + xmax) / 2)
            cy = int((ymin + ymax) / 2)

            objects.append(
                {
                    "id": idx,
                    "pts": points.astype(np.int32),
                    "class": cls,
                    "difficulty": difficulty,
                    "angle": angle,
                    "box": [xmin, ymin, xmax, ymax],
                    "cen": [cx, cy],
                }
            )

    return flight_height, objects


def draw_annotation(img, obj):
    pts = obj["pts"]
    cls = obj["class"]
    difficulty = obj["difficulty"]
    angle = obj["angle"]

    cx, cy = obj["cen"]

    # Difficulty = 1 -> green, otherwise blue
    if difficulty:
        box_color = (0, 255, 0)
    else:
        box_color = (255, 0, 0)

    # Draw OBB
    cv2.polylines(
        img,
        [pts],
        isClosed=True,
        color=box_color,
        thickness=3,
    )

    # Heading arrow
    arrow_length = 40

    arrow_start = [cx, cy]
    arrow_end = [cx + arrow_length, cy]

    arrow = [
        rotate_cart(
            (cx, cy),
            arrow_start,
            math.radians(-angle),
        ),
        rotate_cart(
            (cx, cy),
            arrow_end,
            math.radians(-angle),
        ),
    ]

    cv2.arrowedLine(
        img,
        arrow[0],
        arrow[1],
        (0, 0, 255),
        3,
        tipLength=0.25,
    )

    # Label
    label = f"{cls} | {angle:.1f}"

    text_x = max(0, cx - 40)
    text_y = max(25, cy - 10)

    cv2.putText(
        img,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def main():

    parser = argparse.ArgumentParser(
    description=(
        "Randomly select N images from a DRASHTI-HaOBB dataset split, "
        "visualise the annotated OBBs and heading angles, and generate "
        "individual vehicle crops."
    ),
    formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help=(
            "Dataset split to process.\n"
            "Options: train, val, test"
        ),
    )

    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help=(
            "Number of random images to select and process."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Base name/path for the output directory.\n"
            "The split name is automatically appended.\n"
            "Example: --output Visualise -> Visualise_test/"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed used for image selection "
            "(default: 42)."
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Dataset paths
    # ---------------------------------------------------------

    dataset_dir = "DRASHTI-HaOBB"

    image_dir = os.path.join(
        dataset_dir,
        "images",
        args.split,
    )

    label_dir = os.path.join(
        dataset_dir,
        "labels",
        args.split + "_original",
    )

    # ---------------------------------------------------------
    # Output directory
    # Example:
    # --output Visualise
    # --split test
    #
    # Output:
    # Visualise_test/
    # ---------------------------------------------------------

    out_dir = f"{args.output}_{args.split}"

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        print(f"Removed existing directory: {out_dir}")

    os.makedirs(out_dir, exist_ok=True)

    visual_dir = os.path.join(out_dir, "visual")
    crop_dir = os.path.join(out_dir, "crops")

    os.makedirs(visual_dir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Find images having corresponding labels
    # ---------------------------------------------------------

    valid_images = []

    for filename in os.listdir(image_dir):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            continue

        basename = os.path.splitext(filename)[0]

        label_path = os.path.join(
            label_dir,
            basename + ".txt",
        )

        if os.path.exists(label_path):
            valid_images.append(filename)

    if len(valid_images) == 0:
        raise RuntimeError(
            f"No images with corresponding labels found.\n"
            f"Image directory: {image_dir}\n"
            f"Label directory: {label_dir}"
        )

    # ---------------------------------------------------------
    # Random selection
    # ---------------------------------------------------------

    random.seed(args.seed)

    n_select = min(args.n, len(valid_images))

    selected_images = random.sample(
        valid_images,
        n_select,
    )

    print(f"Split              : {args.split}")
    print(f"Available images   : {len(valid_images)}")
    print(f"Selected images    : {n_select}")
    print(f"Output directory   : {out_dir}")
    print()

    # ---------------------------------------------------------
    # Process selected images
    # ---------------------------------------------------------

    total_crops = 0
    total_objects = 0

    class_counts = {}
    problems = []

    for image_number, image_filename in enumerate(
        selected_images,
        start=1,
    ):

        basename = os.path.splitext(image_filename)[0]

        image_path = os.path.join(
            image_dir,
            image_filename,
        )

        label_path = os.path.join(
            label_dir,
            basename + ".txt",
        )

        img = cv2.imread(image_path)

        if img is None:
            problems.append(
                f"Could not read image: {image_path}"
            )
            continue

        flight_height, objects = read_dota_label(
            label_path
        )

        total_objects += len(objects)

        print(
            f"[{image_number}/{n_select}] "
            f"{image_filename} | "
            f"Objects: {len(objects)} | "
            f"Flight height: {flight_height}"
        )

        # -----------------------------------------------------
        # Annotate every object and save crop
        # -----------------------------------------------------

        for obj in objects:

            cls = obj["class"]
            box = obj["box"]

            class_counts[cls] = (
                class_counts.get(cls, 0) + 1
            )

            # Draw annotation on frame
            draw_annotation(
                img,
                obj,
            )

            xmin, ymin, xmax, ymax = box

            # Ensure valid crop coordinates
            xmin = max(0, min(W - 1, xmin))
            xmax = max(0, min(W, xmax))
            ymin = max(0, min(H - 1, ymin))
            ymax = max(0, min(H, ymax))

            if xmax <= xmin or ymax <= ymin:
                problems.append(
                    f"Invalid crop: "
                    f"{args.split} / "
                    f"{image_filename} / "
                    f"{cls} / {box}"
                )
                continue

            cropped = img[
                ymin:ymax,
                xmin:xmax,
            ]

            if cropped.size == 0:
                problems.append(
                    f"Empty crop: "
                    f"{args.split} / "
                    f"{image_filename} / "
                    f"{cls} / {box}"
                )
                continue

            # -------------------------------------------------
            # Crop directory structure
            #
            # crops/
            #   car/
            #   truck/
            #   2W/
            #   ...
            # -------------------------------------------------

            cls_dir = os.path.join(
                crop_dir,
                cls,
            )

            os.makedirs(
                cls_dir,
                exist_ok=True,
            )

            crop_name = (
                f"{basename}_"
                f"{obj['id']}.jpg"
            )

            crop_path = os.path.join(
                cls_dir,
                crop_name,
            )

            cv2.imwrite(
                crop_path,
                cropped,
            )

            total_crops += 1

        # -----------------------------------------------------
        # Add flight height to visualisation
        # -----------------------------------------------------

        if flight_height is not None:

            cv2.putText(
                img,
                f"Flight Height: {flight_height:.1f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # -----------------------------------------------------
        # Resize frame for visualisation
        # -----------------------------------------------------

        resized = cv2.resize(
            img,
            (1280, 720),
            interpolation=cv2.INTER_AREA,
        )

        visual_path = os.path.join(
            visual_dir,
            image_filename,
        )

        cv2.imwrite(
            visual_path,
            resized,
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"Split              : {args.split}")
    print(f"Selected images    : {n_select}")
    print(f"Total objects      : {total_objects}")
    print(f"Total crops        : {total_crops}")
    print(f"Problems           : {len(problems)}")

    print("\nClass distribution:")

    for cls, count in sorted(
        class_counts.items(),
        key=lambda x: x[0],
    ):
        print(
            f"  {cls:<20} {count}"
        )

    if problems:

        print("\nProblems:")

        for problem in problems:
            print(f"  {problem}")


if __name__ == "__main__":
    main()
