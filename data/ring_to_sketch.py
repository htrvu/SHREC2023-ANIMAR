import argparse
from tqdm import tqdm
import os
import numpy as np
import cv2
import random
from tqdm import trange, tqdm

# random.seed(2001)


def detect_edge(img):
    v = np.median(img)
    sigma = 0.5
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(img, lower, upper)
    return edges


def convert_to_RGB(img):
    rgb_img = 255 - img
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2BGR)
    return rgb_img


def delete_edge_from_pos(img, trace, dx, dy, start_pos, expected_length=40):
    row, col = img.shape
    current_length = 0

    stack = [start_pos]

    while stack:
        cur_pos_x, cur_pos_y = stack.pop()

        trace[cur_pos_x][cur_pos_y] = 1
        img[cur_pos_x][cur_pos_y] = 0
        current_length += 1

        if current_length == expected_length:
            break

        for direct in range(8):
            new_pos_x = cur_pos_x + dx[direct]
            new_pos_y = cur_pos_y + dy[direct]

            if not (0 <= new_pos_x < row and 0 <= new_pos_y < col):
                continue

            if img[new_pos_x, new_pos_y] == 0:
                continue

            if trace[new_pos_x][new_pos_y] == 0:
                stack.append((new_pos_x, new_pos_y))
                break

    return img


def delete_random_edges(img, expected_remove_edges=10, len_per_edge=40):
    copy_img = img.copy()
    row, col = copy_img.shape

    list_pos = np.argwhere(copy_img == 255)
    list_pos = [(x[0], x[1]) for x in list_pos]
    random.shuffle(list_pos)

    trace = np.zeros((row, col), dtype=int)
    dx, dy = [0, 1, 0, -1, 1, 1, -1, -1], [1, 0, -1, 0, 1, -1, 1, -1]

    current_remove_edges = 0
    for pos_x, pos_y in list_pos:
        if current_remove_edges == expected_remove_edges:
            break

        if not trace[pos_x][pos_y]:
            current_remove_edges += 1
            delete_edge_from_pos(copy_img, trace, dx, dy,
                                 (pos_x, pos_y), len_per_edge)

    return copy_img


def convert_to_sketch(img_path, expected_remove_edges, len_per_edge, size_kernel_dialation=3):
    img = cv2.imread(img_path)
    img = detect_edge(img)
    img = delete_random_edges(
        img, expected_remove_edges=expected_remove_edges, len_per_edge=len_per_edge)
    # make image thicker
    kernel = np.ones((size_kernel_dialation, size_kernel_dialation), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = convert_to_RGB(img)
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert images to sketches.')
    parser.add_argument('input', type=str, help='input directory path')
    parser.add_argument('output', type=str, help='output directory path')
    parser.add_argument('--expected_remove_edges', type=int,
                        default=10, help='expected number of edges to remove')
    parser.add_argument('--len_per_edge', type=int,
                        default=30, help='length per edge')
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    expected_remove_edges = args.expected_remove_edges
    len_per_edge = args.len_per_edge

    for ring_type in os.listdir(input_dir):
        ring_path = os.path.join(input_dir, ring_type)
        new_ring_path = os.path.join(output_dir, ring_type)
        print(f"Generating {ring_type}...")
        for obj in tqdm(os.listdir(ring_path)):
            obj_path = os.path.join(ring_path, obj)
            obj_path = os.path.join(obj_path, 'render')

            new_obj_path = os.path.join(new_ring_path, obj)
            new_obj_path = os.path.join(new_obj_path, 'render')

            os.makedirs(new_obj_path, exist_ok=True)

            for img in os.listdir(obj_path):
                img_path = os.path.join(obj_path, img)
                new_img_path = os.path.join(new_obj_path, img)
                sketch_img = convert_to_sketch(
                    img_path, expected_remove_edges, len_per_edge)
                cv2.imwrite(new_img_path, sketch_img)
