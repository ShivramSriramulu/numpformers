import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numpyformer.encoder import NumpyFormerEncoder
from numpyformer.positional import positional_encoding
from exercises.problems import PROBLEMS

st.title("🎮 NumPy-100 Game + NumpyFormer Visualizer")

# Problem selection
problem_idx = st.number_input("Select problem #", min_value=1, max_value=len(PROBLEMS), value=1)
problem = PROBLEMS[problem_idx - 1]

st.subheader(f"Problem {problem_idx}: {problem['desc']}")

# Show example code
if "answer_code" in problem:
    st.write("**Example code:**")
    st.code(problem["answer_code"], language="python")
    
    # Only try to evaluate and visualize if it's not an import statement
    if not problem["answer_code"].startswith("import"):
        try:
            # Create a safe evaluation environment
            safe_dict = {"np": np}
            example_input = eval(problem["answer_code"], safe_dict)
            st.write("**Example output:**")
            st.write(example_input)
            
            # Visualize input if it's an array
            if isinstance(example_input, np.ndarray):
                st.write("**Input Visualization:**")
                fig, ax = plt.subplots()
                if len(example_input.shape) == 1:
                    ax.plot(example_input)
                    ax.set_title("Input 1D Array")
                else:
                    cax = ax.matshow(example_input, cmap='viridis')
                    fig.colorbar(cax)
                    ax.set_title("Input Matrix")
                st.pyplot(fig)
        except Exception as e:
            st.write("Could not evaluate example code:", str(e))

# User code input
user_code = st.text_area("Write your NumPy code:")

# NumpyFormer visualization parameters
st.subheader("🔧 NumpyFormer Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    seq_len = st.slider("Sequence Length", 2, 10, 5)
with col2:
    d_model = st.slider("Model Dimension", 4, 16, 8, step=4)
with col3:
    num_heads = st.slider("Number of Heads", 1, 4, 2)

if d_model % num_heads != 0:
    st.error("Model dimension must be divisible by number of heads")
else:
    # Create random input for NumpyFormer
    x = np.random.randn(1, seq_len, d_model)
    pe = positional_encoding(seq_len, d_model)
    x_pe = x + pe[np.newaxis, :, :]

    # Create encoder and get attention
    encoder = NumpyFormerEncoder(d_model, num_heads)
    _, attn = encoder.forward(x_pe)

    # Visualize attention for each head
    st.subheader("🔍 NumpyFormer Attention Visualization")
    for h in range(attn.shape[1]):
        st.write(f"Head {h+1}")
        fig, ax = plt.subplots()
        cax = ax.matshow(attn[0,h], cmap='viridis')
        fig.colorbar(cax)
        st.pyplot(fig)

if st.button("Run and Compare"):
    try:
        # Run both codes
        user_out = eval(user_code, {"np": np})
        expected_out = eval(problem["answer_code"], {"np": np})

        # Display outputs
        col1, col2 = st.columns(2)
        with col1:
            st.write("🧑‍💻 Your output:")
            st.write(user_out)
            if isinstance(user_out, np.ndarray):
                st.write(f"Shape: {user_out.shape}")
        with col2:
            st.write("✅ Expected output:")
            st.write(expected_out)
            if isinstance(expected_out, np.ndarray):
                st.write(f"Shape: {expected_out.shape}")

        # Compare values
        correct = False
        if isinstance(expected_out, np.ndarray):
            if isinstance(user_out, np.ndarray):
                # For random outputs, only compare shapes
                if "random" in problem["answer_code"].lower():
                    correct = user_out.shape == expected_out.shape
                else:
                    correct = np.allclose(user_out, expected_out)
            else:
                correct = False
        else:
            correct = user_out == expected_out

        if correct:
            st.success("🎉 Correct!")
        else:
            st.warning("❌ Not quite. Let's see how close your structure was...")

        # Ensure numeric array shape
        def to_array(x):
            if isinstance(x, np.ndarray):
                if x.ndim == 1:
                    return x.reshape(1, -1)
                return x
            else:
                return np.array([[x]])

        user_arr = to_array(user_out)
        expected_arr = to_array(expected_out)

        # Pad to same shape
        max_rows = max(user_arr.shape[0], expected_arr.shape[0])
        max_cols = max(user_arr.shape[1], expected_arr.shape[1])
        def pad(arr):
            padded = np.zeros((max_rows, max_cols))
            padded[:arr.shape[0], :arr.shape[1]] = arr
            return padded
        user_arr = pad(user_arr)
        expected_arr = pad(expected_arr)

        # Run both through NumpyFormer
        seq_len, d_model = user_arr.shape
        user_arr = user_arr.reshape(1, seq_len, d_model)
        expected_arr = expected_arr.reshape(1, seq_len, d_model)

        # Add positional encoding
        pe = positional_encoding(seq_len, d_model)
        user_arr = user_arr + pe[np.newaxis, :, :]
        expected_arr = expected_arr + pe[np.newaxis, :, :]

        # Create encoders and get attention
        num_heads = min(2, d_model)  # Ensure d_model is divisible by num_heads
        encoder_user = NumpyFormerEncoder(d_model, num_heads)
        encoder_exp = NumpyFormerEncoder(d_model, num_heads)

        _, attn_user = encoder_user.forward(user_arr)
        _, attn_exp = encoder_exp.forward(expected_arr)

        # Visualize + compare
        st.subheader("🔍 Attention Pattern Comparison")
        for h in range(attn_user.shape[1]):
            st.write(f"Head {h+1} comparison")
            fig, axs = plt.subplots(1,2, figsize=(12,4))
            
            # Plot user attention
            im1 = axs[0].matshow(attn_user[0,h], cmap='viridis')
            axs[0].set_title("Your attention")
            plt.colorbar(im1, ax=axs[0])
            
            # Plot expected attention
            im2 = axs[1].matshow(attn_exp[0,h], cmap='viridis')
            axs[1].set_title("Expected attention")
            plt.colorbar(im2, ax=axs[1])
            
            st.pyplot(fig)

            # Compute similarity score
            diff = np.mean((attn_user[0,h] - attn_exp[0,h])**2)
            st.write(f"Attention similarity (MSE): {diff:.6f}")
            
            # Score the similarity
            if diff < 0.01:
                st.success("🎯 Your attention pattern is very close!")
            elif diff < 0.1:
                st.info("📊 Your attention pattern is somewhat similar")
            else:
                st.warning("⚠️ Your attention pattern differs significantly. Try to match the structure better!")

            # Show attention statistics
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Your attention stats:**")
                st.write(f"Mean: {np.mean(attn_user[0,h]):.3f}")
                st.write(f"Std: {np.std(attn_user[0,h]):.3f}")
                st.write(f"Max: {np.max(attn_user[0,h]):.3f}")
            with col2:
                st.write("**Expected attention stats:**")
                st.write(f"Mean: {np.mean(attn_exp[0,h]):.3f}")
                st.write(f"Std: {np.std(attn_exp[0,h]):.3f}")
                st.write(f"Max: {np.max(attn_exp[0,h]):.3f}")

    except Exception as e:
        st.error(f"Error in your code: {str(e)}")
        st.write("Make sure your code is valid Python and uses NumPy functions correctly.") 