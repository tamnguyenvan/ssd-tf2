"""
"""


def load_label_map(file_path):
    """Load COCO label map from file"""
    label_map = {}
    with open(file_path, 'rt') as f:
        for i, line in enumerate(f):
            label_map[i + 1] = line.strip()
    return label_map
