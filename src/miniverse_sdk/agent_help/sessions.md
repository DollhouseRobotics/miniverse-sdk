# Session evidence and lifecycle

A simulation is ready only after a valid descriptor and monotonically advancing
sequence and physics ticks. For policy bundles also require policy invocation,
actuator write/target evidence, and observable joint, body, or root changes.

Treat health, allocation, `running`, WebSocket connection, and scene rendering
as intermediate signals. Verify command acknowledgement and command-to-effect,
then explicitly stop the session and confirm assignment release. Never stop or
destroy a Vast instance without explicit authorization for that exact instance.
