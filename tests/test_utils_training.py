import pytest
import jax.numpy as jnp
import jax.random as jrnd
from molpipx.utils_training import split_train_and_test_data, split_train_and_test_data_w_forces

@pytest.fixture
def key():
    return jrnd.PRNGKey(42)

def test_split_basic_shapes(key, dataset_context):
    """Test standard split without validation set (Nval=0)."""
    geo, en, _, n_total = dataset_context

    if n_total < 2:
        pytest.skip("Not enough data in Methane.xyz file to test splitting (need > 1 frame)")
    
    n_train = n_total - 1
    
    (X_tr, y_tr), (X_tst, y_tst) = split_train_and_test_data(
        geo, en, N=n_train, key=key, Nval=0
    )
    
    # Check sizes
    assert X_tr.shape[0] == n_train
    assert y_tr.shape[0] == n_train
    assert X_tst.shape[0] == n_total - n_train
    assert y_tst.shape[0] == n_total - n_train
    
    # Check total data is preserved
    assert (X_tr.shape[0] + X_tst.shape[0]) == n_total

    # Check dimensions (Batch, Atoms, 3) Atom = 5 for Methane
    assert X_tr.shape[1:] == (5,3)
    assert X_tst.shape[1:] == (5,3)

def test_split_with_validation(key, dataset_context):
    """
    Test split WITH validation set (Nval > 0).
    """
    geo, en, _, n_total = dataset_context

    if n_total < 3:
        pytest.skip("Not enough data to test split the dataset to Train/val/Test. Need >=3 frames")

    n_train = n_total - 2
    n_val = 1
    
    (X_tr, y_tr), (X_val, y_val) = split_train_and_test_data(
        geo, en, N=n_train, key=key, Nval=n_val
    )
    
    assert X_tr.shape[0] == n_train
    assert y_tr.shape[0] == n_train
    assert X_val.shape[0] == n_val
    assert y_val.shape[0] == n_val
    
    # Ensure no overlap between train and validation sets
    val_sample_geo = X_val[0]
    
    diff = jnp.sum(jnp.abs(X_tr - val_sample_geo), axis = (1,2))
    is_present = jnp.any(diff < 1e-6)

    assert not is_present, "Validation sample found in Training set! Data leakage detected."

def test_split_data_integrity(key, dataset_context):
    """
    CRITICAL: Ensure that Geometry[i] still matches Energy[i] after shuffling.
    """
    geo, en, _, n_total = dataset_context

    if n_total < 1:
        pytest.skip("No data to test integrity.")
    
    
    (X_tr, y_tr), _ = split_train_and_test_data(
        geo, en, N=n_total, key=key
    )
    
    target_geo = geo[0]
    target_en = en[0]

    match_indices = jnp.where(jnp.isclose(y_tr, target_en, atol=1e-5))[0]
    assert len(match_indices) > 0, "Original energy not found in shuffled set!"

    found_pair = False
    for idx in match_indices:
        current_geo = X_tr[idx]
        if jnp.allclose(current_geo, target_geo, atol=1e-5):
            found_pair = True
            break
    
    assert found_pair, "Energy found, but associated Geometry was incorrect! Data mismatch."

def test_split_w_forces_shapes(key, dataset_context):
    """Test the forces version of the split function."""
    geo, en, forces, n_total = dataset_context
    
    if n_total < 2:
        pytest.skip("Not enough data.")

    n_train = n_total - 1
    
    (X_tr, F_tr, y_tr), (X_tst, F_tst, y_tst) = split_train_and_test_data_w_forces(
        geo, forces, en, N=n_train, key=key, Nval=0
    )
    
    assert X_tr.shape[0] == n_train
    assert F_tr.shape[0] == n_train
    assert y_tr.shape[0] == n_train
    
    # Check Integrity: Force value should match Energy value
    assert F_tr.shape == X_tr.shape
    assert F_tst.shape == X_tst.shape

def test_split_w_forces_validation(key, dataset_context):
    """Test forces version with validation set."""
    geo, en, forces, n_total = dataset_context

    if n_total < 3:
        pytest.skip("Not enough data to test split the dataset to Train/val/Test. Need >=3 frames")
    
    n_train = n_total - 2
    n_val = 1
    
    
    (X_tr, F_tr, y_tr), (X_val, F_val, y_val) = split_train_and_test_data_w_forces(
        geo, forces, en, N=n_train, key=key, Nval=n_val
    )
    
    assert F_val.shape[0] == n_val
    assert y_val.shape[0] == n_val
    assert F_val.shape == X_val.shape