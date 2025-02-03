import torch
import torch.nn.functional as F


def compute_ship_placement_maps(board, ship_sizes):
    """
    Compute binary feature maps for each ship size.
    board: (H, W) binary tensor where 1 = unknown, 0 = occupied.
    ship_sizes: List of ship sizes to check.
    Returns a tensor of shape (num_ships, H, W).
    """
    H, W = board.shape
    feature_maps = []

    board_tensor = board.view(1, 1, H, W).float()

    for size in ship_sizes:
        # Create convolution kernels for horizontal and vertical placements
        kernel_h = torch.ones(1, 1, 1, size)  # Horizontal kernel
        kernel_v = torch.ones(1, 1, size, 1)  # Vertical kernel

        # Apply convolution with valid padding and restore shape manually
        valid_h = F.conv2d(F.pad(board_tensor, (size - 1, 0, 0, 0)), kernel_h)
        valid_v = F.conv2d(F.pad(board_tensor, (0, 0, size - 1, 0)), kernel_v)

        # Restore to original shape by trimming extra padding
        valid_h = valid_h[:, :, :, :W]
        valid_v = valid_v[:, :, :H, :]

        # Compute the valid map
        valid_map = ((valid_h.squeeze() == size) | (valid_v.squeeze() == size)).float()
        feature_maps.append(valid_map)

    return torch.stack(feature_maps, dim=0)  # Shape: (num_ships, H, W)


# Example usage
board = torch.ones(5, 5)  # All positions unknown
ship_sizes = [2, 3, 4, 5]
feature_maps = compute_ship_placement_maps(board, ship_sizes)

# Print feature maps to verify correctness
for i, size in enumerate(ship_sizes):
    print(f"Ship Size {size} Feature Map:\n{feature_maps[i].int()}\n")
