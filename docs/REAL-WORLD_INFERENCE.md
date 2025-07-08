# Inference
Seer has been encapsulated into a real-world controller, allowing for straightforward adaptation to downstream tasks. To ensure smooth implementation and avoid common errors, we offer the following recommendations:
* :fire: **Proprio First v.s. Image First:** During data collection and inference, always acquire robot proprioception data first, followed by image observations. This order minimizes the timestep interval since capturing images is significantly more time-intensive than reading proprioception data.
* :fire: **Delta-to-Absolute Action Labels:** To simplify matrix computation and transformation, we provide a delta-action-to-absolute-action conversion in the [deployment script](../deploy.py). This aligns with the absolute-action-to-delta-action transformation found in the [post-process script](../utils/real_ft_data.py).
* :fire: **Consistent Control Frequency:** Ensure that the control frequencies used during data collection match those during inference. Discrepancies in frequency can lead to inconsistent results.

## :star: Real-World Controller
A [wrapped seer controller](../real_controller/controller.py) is provided for real-world deployment. This controller is modular and can be easily adapted to specific tasks or environments.

## :star2: Real-World Deployment Pseudocode
To deploy the wrapped Seer controller for real-world tasks, modify the [deployment script](../deploy.py) to fit your specific environment. Then, execute the deployment with the following command:
```python
bash scripts/REAL/deploy.sh
```