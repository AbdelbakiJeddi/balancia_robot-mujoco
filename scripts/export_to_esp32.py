"""
Export trained Stable-Baselines3 PPO policy to a lightweight, standalone C/C++ header
that can run directly on ESP32 / ESP32-S3 / Arduino without any external ML dependencies.

Usage:
    python scripts/export_to_esp32.py --model trained_models/ppo_balance_latest.zip --output-dir esp32_firmware
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO


def export_policy_to_c_header(model_path: Path, output_file: Path):
    print(f"Loading model from {model_path} ...")
    model = PPO.load(str(model_path), device="cpu")
    policy = model.policy

    # Extract policy (actor) MLP weights
    # In SB3 MlpPolicy: policy.mlp_extractor.policy_net contains hidden layers,
    # policy.action_net contains the final layer to action mean.
    weights = []
    biases = []

    # Hidden layers
    for layer in policy.mlp_extractor.policy_net:
        if isinstance(layer, torch.nn.Linear):
            weights.append(layer.weight.detach().cpu().numpy())
            biases.append(layer.bias.detach().cpu().numpy())

    # Final action layer (mean)
    action_layer = policy.action_net
    weights.append(action_layer.weight.detach().cpu().numpy())
    biases.append(action_layer.bias.detach().cpu().numpy())

    num_layers = len(weights)
    print(f"Extracted {num_layers} policy layers:")
    for i, (w, b) in enumerate(zip(weights, biases)):
        print(f"  Layer {i}: in_features={w.shape[1]}, out_features={w.shape[0]}")

    in_dim = weights[0].shape[1]
    out_dim = weights[-1].shape[0]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write("/*\n")
        f.write(" * Auto-generated neural network policy weights for ESP32 / Arduino\n")
        f.write(f" * Input dimension  : {in_dim}\n")
        f.write(f" * Output dimension : {out_dim}\n")
        f.write(f" * Architecture     : {[w.shape[1] for w in weights]} -> {out_dim}\n")
        f.write(" * Activation       : Tanh (hidden layers), Linear (output)\n")
        f.write(" */\n\n")
        f.write("#ifndef POLICY_NET_H\n")
        f.write("#define POLICY_NET_H\n\n")
        f.write("#include <math.h>\n\n")

        # Layer dimensions
        f.write(f"#define NN_INPUT_DIM  {in_dim}\n")
        f.write(f"#define NN_OUTPUT_DIM {out_dim}\n\n")

        # Write weights and biases as C arrays
        for i, (w, b) in enumerate(zip(weights, biases)):
            rows, cols = w.shape
            f.write(f"// Layer {i} Weights [{rows}][{cols}]\n")
            f.write(f"static const float W{i}[{rows}][{cols}] = {{\n")
            for r in range(rows):
                row_str = ", ".join(f"{val:.8f}f" for val in w[r])
                f.write(f"    {{ {row_str} }},\n")
            f.write("};\n\n")

            f.write(f"// Layer {i} Biases [{rows}]\n")
            bias_str = ", ".join(f"{val:.8f}f" for val in b)
            f.write(f"static const float B{i}[{rows}] = {{ {bias_str} }};\n\n")

        # Write the inference forward pass function
        f.write("/**\n")
        f.write(" * Fast forward inference pass on microcontroller (pure C, zero dependencies).\n")
        f.write(" * @param obs Input observation array [pitch, pitch_rate, forward_vel, lv, rv, target_vel]\n")
        f.write(" * @param action Output action array [u_balance] in [-1.0, 1.0]\n")
        f.write(" */\n")
        f.write("static inline void policy_predict(const float obs[NN_INPUT_DIM], float action[NN_OUTPUT_DIM]) {\n")

        # Hidden layer buffers
        for i in range(num_layers - 1):
            dim = weights[i].shape[0]
            f.write(f"    float h{i}[{dim}];\n")

        f.write("\n")

        # Layer 0 forward
        w0_rows, w0_cols = weights[0].shape
        f.write(f"    // Forward Layer 0 (Input -> Hidden 0)\n")
        f.write(f"    for (int i = 0; i < {w0_rows}; i++) {{\n")
        f.write(f"        float sum = B0[i];\n")
        f.write(f"        for (int j = 0; j < {w0_cols}; j++) {{\n")
        f.write(f"            sum += W0[i][j] * obs[j];\n")
        f.write(f"        }}\n")
        f.write(f"        h0[i] = tanhf(sum);\n")
        f.write(f"    }}\n\n")

        # Intermediate hidden layers
        for l in range(1, num_layers - 1):
            wl_rows, wl_cols = weights[l].shape
            prev_l = l - 1
            f.write(f"    // Forward Layer {l}\n")
            f.write(f"    for (int i = 0; i < {wl_rows}; i++) {{\n")
            f.write(f"        float sum = B{l}[i];\n")
            f.write(f"        for (int j = 0; j < {wl_cols}; j++) {{\n")
            f.write(f"            sum += W{l}[i][j] * h{prev_l}[j];\n")
            f.write(f"        }}\n")
            f.write(f"        h{l}[i] = tanhf(sum);\n")
            f.write(f"    }}\n\n")

        # Final layer forward (Linear output)
        last_l = num_layers - 1
        w_last_rows, w_last_cols = weights[last_l].shape
        prev_last = last_l - 1
        f.write(f"    // Output Layer (Hidden {prev_last} -> Action)\n")
        f.write(f"    for (int i = 0; i < {w_last_rows}; i++) {{\n")
        f.write(f"        float sum = B{last_l}[i];\n")
        f.write(f"        for (int j = 0; j < {w_last_cols}; j++) {{\n")
        f.write(f"            sum += W{last_l}[i][j] * h{prev_last}[j];\n")
        f.write(f"        }}\n")
        f.write(f"        // Clip action to [-1.0, 1.0]\n")
        f.write(f"        if (sum > 1.0f) sum = 1.0f;\n")
        f.write(f"        if (sum < -1.0f) sum = -1.0f;\n")
        f.write(f"        action[i] = sum;\n")
        f.write(f"    }}\n")
        f.write("}\n\n")
        f.write("#endif // POLICY_NET_H\n")

    print(f"Successfully generated C/C++ header: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Export trained PPO policy to C/C++ header for ESP32")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parents[1] / "trained_models" / "ppo_balance_latest.zip"),
        help="Path to trained PPO .zip file",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "firmware" / "balancia_esp32_s3_zero"),
        help="Directory to save policy_net.h",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    output_file = Path(args.output_dir) / "policy_net.h"
    export_policy_to_c_header(model_path, output_file)


if __name__ == "__main__":
    main()
