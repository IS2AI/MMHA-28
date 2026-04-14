import os

def rename_rgb_frames(folder_path):
    files = sorted(f for f in os.listdir(folder_path) if f.endswith('.png') and f.startswith('rgb_'))
    for i, old_name in enumerate(files, 1):
        new_name = f"rgb_{i:04d}.png"
        os.rename(os.path.join(folder_path, old_name), os.path.join(folder_path, new_name))

# Recursively rename all rgb_images folders
for root, dirs, _ in os.walk('dataset/val/session_1/sub_2/d_rgb/1/rgb_images'):
    print(f"Renaming in: {root}")
    rename_rgb_frames(root)
