#! pip install lightning-sdk
import os
import time
import uuid

from lightning_sdk import Machine, Studio


scripts = [
    ("pytorch_lightning/ptl_default.py", "", ["ptl_default.ckpt"]),
    ("pytorch_lightning/ptl_ddp.py", "", ["ptl_ddp.ckpt"]),
    ("pytorch_lightning/ptl_fsdp.py", "", ["ptl_fsdp.ckpt"]),
    ("pytorch_lightning/ptl_deepspeed.py", "deepspeed", ["ptl_deepspeed.ckpt"]),
    ("fabric/fabric_default.py", "", ["fabric_default.ckpt"]),
    ("fabric/fabric_ddp.py", "", ["fabric_ddp.ckpt"]),
    ("fabric/fabric_fsdp.py", "", ["fabric_fsdp.ckpt"]),
    ("fabric/fabric_deepspeed.py", "deepspeed", ["fabric_deepspeed.ckpt"]),
    ("pytorch/pytorch_ddp.py", "", ["pytorch_ddp.ckpt"]),
]


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

    job_names_and_artifacts = []

    for script, requirements, artifacts in scripts:
        entrypoint = os.path.join(workdir, script)
        script_name, _ = os.path.splitext(os.path.basename(script))
        job_name = f"mmt-template-test-{prefix}-{script_name}"
        job_names_and_artifacts.append(job_name, artifacts)

        print(f"Starting MMT job {job_name}: {entrypoint}")
        plugin.run(
            command=f"pip install {requirements} && python {entrypoint}",
            name=job_name,
            cloud_compute=Machine.T4,
            num_instances=2
        )

    studio.stop()

    timeout = 15 * 60  # 15 minutes
    t0 = time.time()

    missing_artifacts = []

    while True:
        t1 = time.time()
        if t1 - t0 > timeout:
            break
        missing_artifacts = []
        for name, artifacts in job_names_and_artifacts:
            # TODO: also verify size or content
            for artifact in artifacts:
                expected = f"/teamspace/jobs/{name}/{artifact}"
                if not os.path.exists(expected):
                    missing_artifacts.append(expected)
        if not missing_artifacts:
            break

    assert not missing_artifacts, f"Artifacts still missing after {timeout}s: {' '.join(missing_artifacts)}"


if __name__ == "__main__":
    main()
