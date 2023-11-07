#! pip install lightning-sdk
import os
import time
import uuid

from lightning_sdk import Machine, Studio


def main():
    prefix = str(uuid.uuid4())[:8]
    studio_name = f"mmt-template-test-{prefix}-studio"

    print(f"Starting Studio {studio_name}")

    plugin_name = "multi-machine-training"

    studio = Studio(studio_name, teamspace="thunder", create_ok=True)
    studio.start()

    studio.install_plugin(plugin_name)
    plugin = studio.installed_plugins[plugin_name]

    studio.run(
        "git clone https://github.com/Lightning-AI/Lightning-multinode-templates.git &&"
        "cd Lightning-multinode-templates && git checkout pl-multinode"
    )

    workdir = "Lightning-multinode-templates"

    scripts = [
        "pytorch_lightning/ptl_default.py",
        "pytorch_lightning/ptl_ddp.py",
        "pytorch_lightning/ptl_fsdp.py",
        "pytorch_lightning/ptl_deepspeed.py",
        "fabric/fabric_default.py",
        "fabric/fabric_ddp.py",
        "fabric/fabric_fsdp.py",
        "fabric/fabric_deepspeed.py",
        "pytorch/pytorch_dpp.py"
    ]

    names_to_artifacts = {}

    for script in scripts:
        entrypoint = os.path.join(workdir, script)
        script_name, _ = os.path.splitext(os.path.basename(script))
        name = f"mmt-template-test-{prefix}-{script_name}"

        print(f"Starting MMT job {name}: {entrypoint}")
        plugin.run(
            command=f"python {entrypoint}",
            name=name,
            cloud_compute=Machine.T4,
            num_instances=2
        )

        names_to_artifacts[name] = f"{script_name}.ckpt"

    studio.stop()

    timeout = 15 * 60  # 15 minutes
    t0 = time.time()

    missing_artifacts = []

    while True:
        t1 = time.time()
        if t1 - t0 > timeout:
            break
        missing_artifacts = []
        for name, artifact in names_to_artifacts.items():
            # TODO: also verify size or content
            expected = f"/teamspace/jobs/{name}/{artifact}"
            if not os.path.exists(expected):
                missing_artifacts.append(expected)
        if not missing_artifacts:
            break

    assert not missing_artifacts, f"Artifacts still missing after {timeout}s: {' '.join(missing_artifacts)}"


if __name__ == "__main__":
    main()
