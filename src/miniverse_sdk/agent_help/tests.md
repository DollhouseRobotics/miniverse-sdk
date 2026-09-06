# Approved tests

Run a UTF-8 Python test against an immutable bundle revision:

```bash
miniverse test start ID@REVISION_ID --file test_policy.py --seed 0 --json
miniverse test start ID@REVISION_ID --file - --no-wait --json
miniverse test status SESSION_ID --json
miniverse test results SESSION_ID --wait --timeout 300 --json
miniverse test stop SESSION_ID --json
```

`start` waits for results unless `--no-wait` is used. Source must be nonempty
UTF-8 and at most 64 KiB. `--seed` defaults to `0`; each invocation generates a
UUID idempotency key unless `--idempotency-key` is supplied. A client polling
timeout returns JSON containing the `sessionId`, so results can be resumed.
Reports are retained for 24 hours. Outcomes are `passed`, `assertion_failed`,
`policy_failed`, `test_error`, `timed_out`, `cancelled`, or
`infrastructure_failed`.

Exit status `0` means passed (or a successful non-result operation), `4` means
a nonpassing assertion, policy, or test report, and `5` means a timed-out,
cancelled, or infrastructure-failed report or client polling timeout. Existing
local, validation, and API failures retain statuses `1`, `2`, and `3`.

## Test program contract

Define `class Test` with `step(self, frame)`. Return
`TestContinue(commands={})` to advance one policy step (optionally replacing
declared command values), or `TestDone(result={})` to finish with a bounded,
JSON-compatible result. The test sandbox provides restricted `np` math and the
test interfaces; it does not provide arbitrary imports, filesystem, or network
access. The bundle policy runs in a separate sandbox instance with the same
restrictions, so test state cannot access policy state.

`frame.sim_data` is canonical backend-neutral state captured after policy
actuation is committed. It has `physics_tick`, `simulation_time`,
`joint_positions`, `joint_velocities`, `joint_efforts`, `body_transforms`,
`body_velocities`, optional `contacts`, and immutable `model`. `model` exposes
`backend`, `version`, ordered `body_ids`, `joint_ids`, `actuator_ids`,
`actuator_joint_ids`, `root_body_id`, joint widths, DOF data, body dynamics and
masses, and terrains. Contacts expose ordered `body_ids`, `sample_dt`, world
net forces and impulses, and backend-supported normal forces and pair counts.
`frame.evidence` contains bounded runtime evidence for that committed step,
including `acceptedActuation`, `effectiveCommands`, and `cadence` counters.

For example, check finite joint state over one simulated second:

```python
class Test:
    def step(self, frame):
        data = frame.sim_data
        assert np.isfinite(data.joint_positions).all(), "Non-finite joints"
        if data.simulation_time >= 1.0:
            return TestDone(result={"physics_tick": data.physics_tick})
        return TestContinue()
```

Use persistent instance attributes for temporal measurements. Return
`TestContinue(commands={"your-command-id": value})` to change a declared command
at the next policy boundary. Physical effects appear in subsequent states.
Unavailable backend state raises an error rather than supplying fake zeros.
Each callback has a 250 ms ceiling; initialization has a five-second ceiling.
Both count against the overall wall-clock deadline. Returned JSON and the
complete report must fit within 64 KiB.

Tests receive at most 30 seconds of active execution wall time. Time spent in
the admission queue or worker provisioning does not consume that budget.
There is no automatic restart after activation. Existing OAuth users should
run `miniverse auth login` again if their grant lacks the `read` and `write` permissions;
personal API tokens retain their existing account permissions. Treat a passing report as
the only success signal; allocation, running status, or a connected worker is
not test evidence.
