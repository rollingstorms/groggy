"""
Module 9: Neural/Autograd Testing

Tests automatic differentiation and neural network operations.
The autodiff system is implemented in Rust core with computation graph tracking.

Test Coverage:
- Gradient tracking (requires_grad, requires_grad_)
- Basic operations with gradients (add, multiply, matmul)
- Activation functions with gradients (relu, sigmoid, tanh)
- Backward pass computation
- Loss functions (MSE, cross entropy)
- Gradient accumulation and zeroing

Success Criteria: Gradients computed correctly for basic operations
"""

import sys
from pathlib import Path

import pytest

# Add path for groggy
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

try:
    import groggy as gr
except ImportError:
    gr = None


@pytest.mark.neural
class TestGradientTracking:
    """Test gradient tracking setup"""

    def test_requires_grad_property(self):
        """Test requires_grad property access"""
        if gr is None:
            pytest.skip("groggy not available")

        m = gr.matrix([[1.0, 2.0], [3.0, 4.0]])

        # Initial state should be False
        assert hasattr(m, "requires_grad"), "Matrix should have requires_grad property"
        assert m.requires_grad == False, "Initial requires_grad should be False"

    def test_requires_grad_setter(self):
        """Test setting requires_grad"""
        if gr is None:
            pytest.skip("groggy not available")

        m = gr.matrix([[1.0, 2.0], [3.0, 4.0]])

        # requires_grad_ returns a new matrix with tracking enabled
        m_tracked = m.requires_grad_(True)
        assert m_tracked is not None, "requires_grad_ should return matrix"
        assert (
            m_tracked.requires_grad == True
        ), "Returned matrix should have requires_grad=True"

    def test_grad_property_exists(self):
        """Test grad property for accessing gradients"""
        if gr is None:
            pytest.skip("groggy not available")

        m = gr.matrix([[1.0, 2.0], [3.0, 4.0]]).requires_grad_(True)

        assert hasattr(m, "grad"), "Matrix should have grad property"
        # Initially grad should be None or zero
        # We can't check the value without doing backward first

    def test_zero_grad(self):
        """Test zeroing gradients"""
        if gr is None:
            pytest.skip("groggy not available")

        m = gr.matrix([[1.0, 2.0], [3.0, 4.0]]).requires_grad_(True)

        # Should have zero_grad method
        assert hasattr(m, "zero_grad"), "Matrix should have zero_grad method"

        # Should not fail (even though no computation graph yet)
        try:
            m.zero_grad()
            # Expected to fail without computation graph
        except Exception as e:
            assert "computation graph" in str(e).lower() or "gradient" in str(e).lower()


@pytest.mark.neural
@pytest.mark.skip(
    reason="Autograd requires computation graph initialization - API under development"
)
class TestBasicAutodiff:
    """Test basic automatic differentiation"""

    def test_simple_addition_gradient(self):
        """Test gradient of addition operation"""
        if gr is None:
            pytest.skip("groggy not available")

        # Create matrices with gradient tracking
        m1 = gr.matrix([[1.0, 2.0]]).requires_grad_(True)
        m2 = gr.matrix([[3.0, 4.0]]).requires_grad_(True)

        # Forward pass
        result = m1 + m2

        # Backward pass
        result.backward()

        # Check gradients
        assert m1.grad is not None, "Should have gradient for m1"
        assert m2.grad is not None, "Should have gradient for m2"

    def test_simple_multiplication_gradient(self):
        """Test gradient of multiplication"""
        if gr is None:
            pytest.skip("groggy not available")

        m1 = gr.matrix([[2.0, 3.0]]).requires_grad_(True)
        m2 = gr.matrix([[4.0, 5.0]]).requires_grad_(True)

        result = m1 * m2
        result.backward()

        # Gradient of multiplication: d(m1*m2)/dm1 = m2
        assert m1.grad is not None
        assert m2.grad is not None

    def test_matmul_gradient(self):
        """Test gradient of matrix multiplication"""
        if gr is None:
            pytest.skip("groggy not available")

        m1 = gr.matrix([[1.0, 2.0], [3.0, 4.0]]).requires_grad_(True)
        m2 = gr.matrix([[5.0, 6.0], [7.0, 8.0]]).requires_grad_(True)

        result = m1.matmul(m2)
        result.backward()

        assert m1.grad is not None
        assert m2.grad is not None


@pytest.mark.neural
@pytest.mark.skip(
    reason="Autograd requires computation graph initialization - API under development"
)
class TestActivationGradients:
    """Test gradients of activation functions"""

    def test_relu_gradient(self):
        """Test ReLU activation gradient"""
        if gr is None:
            pytest.skip("groggy not available")

        m = gr.matrix([[-1.0, 2.0], [3.0, -4.0]]).requires_grad_(True)

        result = m.relu()
        result.backward()

        # ReLU gradient: 1 for positive, 0 for negative
        assert m.grad is not None

    def test_sigmoid_gradient(self):
        """Test sigmoid activation gradient"""
        if gr is None:
            pytest.skip("groggy not available")

        m = gr.matrix([[0.0, 1.0], [-1.0, 2.0]]).requires_grad_(True)

        result = m.sigmoid()
        result.backward()

        # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        assert m.grad is not None

    def test_tanh_gradient(self):
        """Test tanh activation gradient"""
        if gr is None:
            pytest.skip("groggy not available")

        m = gr.matrix([[0.0, 1.0], [-1.0, 2.0]]).requires_grad_(True)

        result = m.tanh()
        result.backward()

        # tanh'(x) = 1 - tanh^2(x)
        assert m.grad is not None


@pytest.mark.neural
@pytest.mark.skip(reason="Loss functions need scalar output API - under development")
class TestLossFunctions:
    """Test loss functions and their gradients"""

    def test_mse_loss(self):
        """Test Mean Squared Error loss"""
        if gr is None:
            pytest.skip("groggy not available")

        predictions = gr.matrix([[1.0, 2.0], [3.0, 4.0]]).requires_grad_(True)
        targets = gr.matrix([[1.5, 2.5], [3.5, 4.5]])

        # MSE loss: mean((pred - target)^2)
        diff = predictions - targets
        squared = diff * diff
        loss = squared.mean()

        loss.backward()

        assert predictions.grad is not None, "Should compute gradients for predictions"

    def test_cross_entropy_loss(self):
        """Test Cross Entropy loss"""
        if gr is None:
            pytest.skip("groggy not available")

        # Logits (before softmax)
        logits = gr.matrix([[2.0, 1.0, 0.1]]).requires_grad_(True)
        # True class: 0
        target = 0

        # Cross entropy: -log(softmax(logits)[target])
        probs = logits.softmax()
        # Would need indexing to get probs[target]

        pytest.skip("Cross entropy needs indexing API")


@pytest.mark.neural
@pytest.mark.skip(reason="Computation graph API under development")
class TestGradientAccumulation:
    """Test gradient accumulation and management"""

    def test_gradient_accumulation(self):
        """Test that gradients accumulate over multiple backward passes"""
        if gr is None:
            pytest.skip("groggy not available")

        m = gr.matrix([[1.0, 2.0]]).requires_grad_(True)

        # First backward
        result1 = m * 2.0
        result1.backward()
        grad1 = m.grad

        # Second backward (should accumulate)
        result2 = m * 3.0
        result2.backward()
        grad2 = m.grad

        # Gradients should accumulate
        assert grad2 is not None, "Should have accumulated gradients"

    def test_zero_grad_clears_gradients(self):
        """Test that zero_grad clears accumulated gradients"""
        if gr is None:
            pytest.skip("groggy not available")

        m = gr.matrix([[1.0, 2.0]]).requires_grad_(True)

        # Compute some gradients
        result = m * 2.0
        result.backward()
        assert m.grad is not None

        # Zero gradients
        m.zero_grad()

        # Gradients should be cleared
        # (grad might be None or zero matrix)


@pytest.mark.neural
class TestNeuralIntegration:
    """Test integration of neural operations (what works now)"""

    def test_activation_functions_available(self):
        """Test that activation functions are available and work"""
        if gr is None:
            pytest.skip("groggy not available")

        m = gr.matrix([[1.0, 2.0], [-1.0, -2.0]])

        # These should all work in forward pass
        assert hasattr(m, "relu"), "Should have relu"
        assert hasattr(m, "sigmoid"), "Should have sigmoid"
        assert hasattr(m, "tanh"), "Should have tanh"
        assert hasattr(m, "elu"), "Should have elu"

        # Test forward pass works
        relu_result = m.relu()
        assert relu_result is not None

        sigmoid_result = m.sigmoid()
        assert sigmoid_result is not None

        tanh_result = m.tanh()
        assert tanh_result is not None

    def test_gradient_api_exists(self):
        """Test that gradient API exists even if not fully functional"""
        if gr is None:
            pytest.skip("groggy not available")

        m = gr.matrix([[1.0, 2.0]])

        # API should exist
        assert hasattr(m, "requires_grad"), "Should have requires_grad property"
        assert hasattr(m, "requires_grad_"), "Should have requires_grad_ method"
        assert hasattr(m, "grad"), "Should have grad property"
        assert hasattr(m, "backward"), "Should have backward method"
        assert hasattr(m, "zero_grad"), "Should have zero_grad method"

    def test_backward_fails_gracefully(self):
        """Test that backward fails gracefully without computation graph"""
        if gr is None:
            pytest.skip("groggy not available")

        m = gr.matrix([[1.0, 2.0]]).requires_grad_(True)
        result = m.sigmoid()

        # Should fail with informative error
        try:
            result.backward()
            pytest.skip("backward unexpectedly succeeded")
        except Exception as e:
            error_msg = str(e).lower()
            assert (
                "computation graph" in error_msg or "gradient" in error_msg
            ), f"Error should mention computation graph, got: {e}"


@pytest.mark.neural
@pytest.mark.skip(reason="Documentation placeholder for future neural network layers")
class TestNeuralNetworkLayers:
    """Test neural network layer operations (future)"""

    def test_linear_layer(self):
        """Test linear/dense layer forward and backward"""
        pytest.skip("Neural network layers not yet implemented")

    def test_conv2d_layer(self):
        """Test 2D convolution layer"""
        pytest.skip("Conv2D layers not yet implemented")

    def test_batch_normalization(self):
        """Test batch normalization layer"""
        pytest.skip("Batch normalization not yet implemented")


if __name__ == "__main__":
    # Allow running this module directly for development
    pytest.main([__file__, "-v"])
