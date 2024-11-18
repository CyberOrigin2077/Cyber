# CYBER World Model

Imagine a robot navigating a cluttered room. For it to operate effectively, it needs to "see" beyond raw images—recognizing objects, understanding spatial relationships, and predicting interactions. Physical Encoding achieves this by converting the robot’s visual input into latent codes. These codes represent a distilled understanding of the physical world, akin to how humans quickly grasp the layout of a room upon entering. For example, if the robot sees a chair, the encoded information isn't just "chair" but also its position, how it might be moved, and its potential use in future tasks. This high-level encoding enables the robot to focus on important environmental features and make smarter decisions.

## Components of a World Model
Cyber world models consists of two main components: an **autoencoder** and a **dynamic model**. The autoencoder learns to encode raw observations into a compact representation, while the dynamic model predicts future states in this representation.

## Getting Started
Interfaces for each component are standardized and cross-compatible, allowing you to mix and match different models. This flexibility enables you to experiment with various architectures and training strategies, tailoring the world model to your specific needs.
```python
from cyber.models.world import Autoencoder, DynamicModel
import cyber.world as world

# Load the Autoencoder
autoencoder:Autoencoder = world.autoencoder.VQModel.from_pretrained("path/to/autoencoder/weights")
# Load the Dynamic Model
dynamic_model:DynamicModel = world.dynamic.STMaskGIT.from_pretrained("path/to/dynamic/model/weights")

# Set the models to evaluation mode for inference
autoencoder.eval()
dynamic_model.eval()

# Example input data
input_obs = torch.randn(1, 3, 64, 64) # (batch_size, channels, height, width)

# Use the autoencoder to encode the input data
input_enc = autoencoder.encode(input_obs)

# Use the dynamic model to predict the next state
predicted_enc = dynamic_model(encoded_obs)

# Decode back into observation space
predicted_obs = autoencoder.decode(predicted_obs)
```