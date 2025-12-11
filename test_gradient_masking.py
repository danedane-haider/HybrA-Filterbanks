"""Test script to verify gradient masking works correctly for learnable ISAC kernels."""

import torch
import torch.nn as nn
from hybra.isac import ISAC


def test_gradient_masking():
    """Verify that gradients are only computed for non-zero kernel elements."""

    print("="*70)
    print("Testing Gradient Masking for Learnable ISAC Kernels")
    print("="*70)

    # Create ISAC filterbank with learnable encoder
    print("\n1. Creating ISAC filterbank with learnable encoder...")
    filterbank = ISAC(
        fs=16000,
        kernel_size=128,
        num_channels=40,
        L=16000,
        is_encoder_learnable=True,
        verbose=False
    )

    # Check that mask exists
    print("   ✓ Filterbank created")
    assert hasattr(filterbank, 'kernel_mask'), "kernel_mask buffer not found!"
    print("   ✓ Gradient mask registered")

    # Verify mask shape matches kernels
    assert filterbank.kernel_mask.shape == filterbank.kernels.shape, \
        f"Mask shape {filterbank.kernel_mask.shape} != kernel shape {filterbank.kernels.shape}"
    print(f"   ✓ Mask shape matches kernels: {filterbank.kernel_mask.shape}")

    # Check mask values are binary (0 or 1)
    unique_vals = torch.unique(filterbank.kernel_mask)
    assert torch.all((unique_vals == 0) | (unique_vals == 1)), \
        f"Mask contains non-binary values: {unique_vals}"
    print(f"   ✓ Mask is binary with {torch.sum(filterbank.kernel_mask == 0).item()} zeros and {torch.sum(filterbank.kernel_mask == 1).item()} ones")

    # Verify mask matches kernel structure
    kernel_nonzero = (torch.abs(filterbank.kernels.data) != 0).float()
    assert torch.allclose(filterbank.kernel_mask, kernel_nonzero), \
        "Mask doesn't match actual kernel zero pattern!"
    print("   ✓ Mask correctly identifies non-zero elements")

    print("\n2. Testing gradient computation...")

    # Create dummy input and run forward pass
    x = torch.randn(2, 16000)

    # Store original kernel values for zero-padded positions
    zero_positions = filterbank.kernel_mask == 0
    original_zeros = filterbank.kernels.data[zero_positions].clone()

    # Forward pass
    coeffs = filterbank(x)

    # Compute loss and backward
    loss = coeffs.mean()
    loss.backward()

    print("   ✓ Forward and backward pass completed")

    # Check that gradients exist
    assert filterbank.kernels.grad is not None, "No gradients computed!"
    print("   ✓ Gradients computed")

    # Verify gradients are zero where mask is zero
    masked_grads = filterbank.kernels.grad[zero_positions]
    assert torch.allclose(masked_grads, torch.zeros_like(masked_grads), atol=1e-10), \
        f"Gradients not zero for masked positions! Max grad: {masked_grads.abs().max()}"
    print(f"   ✓ Gradients are zero for {zero_positions.sum().item()} zero-padded positions")

    # Verify gradients are non-zero where mask is one (at least some of them)
    nonzero_positions = filterbank.kernel_mask == 1
    nonzero_grads = filterbank.kernels.grad[nonzero_positions]
    num_nonzero_grads = (nonzero_grads.abs() > 1e-10).sum().item()
    assert num_nonzero_grads > 0, "No non-zero gradients found for non-masked positions!"
    print(f"   ✓ {num_nonzero_grads}/{nonzero_positions.sum().item()} non-masked positions have non-zero gradients")

    print("\n3. Testing optimizer step...")

    # Run optimizer step
    optimizer = torch.optim.SGD([filterbank.kernels], lr=0.01)
    optimizer.step()

    # Verify zero-padded positions remain exactly zero
    updated_zeros = filterbank.kernels.data[zero_positions]
    assert torch.allclose(updated_zeros, original_zeros, atol=1e-10), \
        f"Zero-padded positions changed! Max change: {(updated_zeros - original_zeros).abs().max()}"
    print(f"   ✓ Zero-padded positions remain exactly zero after optimizer step")

    # Verify non-zero positions were updated
    nonzero_vals_before = filterbank.aud_kernels[nonzero_positions]
    nonzero_vals_after = filterbank.kernels.data[nonzero_positions]
    num_changed = (torch.abs(nonzero_vals_after - nonzero_vals_before) > 1e-10).sum().item()
    assert num_changed > 0, "No non-zero positions were updated!"
    print(f"   ✓ {num_changed}/{nonzero_positions.sum().item()} non-zero positions were updated")

    print("\n4. Statistics:")
    print(f"   - Total kernel elements: {filterbank.kernels.numel()}")
    print(f"   - Zero-padded elements: {zero_positions.sum().item()} ({100*zero_positions.sum().item()/filterbank.kernels.numel():.1f}%)")
    print(f"   - Learnable elements: {nonzero_positions.sum().item()} ({100*nonzero_positions.sum().item()/filterbank.kernels.numel():.1f}%)")

    # Show per-channel statistics
    print("\n5. Per-channel zero-padding:")
    for i in range(min(5, filterbank.kernels.shape[0])):  # Show first 5 channels
        channel_zeros = (filterbank.kernel_mask[i] == 0).sum().item()
        channel_total = filterbank.kernel_mask[i].numel()
        print(f"   Channel {i:2d}: {channel_zeros:4d}/{channel_total} zeros ({100*channel_zeros/channel_total:5.1f}%)")
    if filterbank.kernels.shape[0] > 5:
        print(f"   ... (showing 5/{filterbank.kernels.shape[0]} channels)")

    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70)
    print("\nGradient masking is working correctly:")
    print("  • Mask correctly identifies zero-padded regions")
    print("  • Gradients are zeroed for zero-padded positions")
    print("  • Only non-zero kernel elements are updated during training")
    print("  • Zero-padded regions remain exactly zero after optimization")


if __name__ == "__main__":
    test_gradient_masking()
