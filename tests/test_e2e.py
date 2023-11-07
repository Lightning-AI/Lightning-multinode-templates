#! pip install lightning-sdk
import os
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

    # TODO:
    # - verify artifacts
    studio.stop()


if __name__ == "__main__":
    main()