#! pip install lightning-sdk
import json
import os
import time
import uuid

from lightning_sdk import Machine, Studio


def main(recipe: dict | str, prefix: str = "", timeout: int = 15 * 60):
    if isinstance(recipe, str):
        recipe = json.loads(recipe)

    script = recipe["script"]
    requirements = recipe["requirements"]
    artifacts = recipe["artifacts"]

    prefix = prefix or str(uuid.uuid4())[:8]
    studio_name = f"mmt-template-test-{prefix}-studio"

    print(f"Starting Studio {studio_name}")

    plugin_name = "multi-machine-training"

    studio = Studio(studio_name, create_ok=True)
    studio.start()

    studio.install_plugin(plugin_name)
    plugin = studio.installed_plugins[plugin_name]

    studio.run(
        "git clone https://github.com/Lightning-AI/Lightning-multinode-templates.git &&"
        "cd Lightning-multinode-templates && git checkout pl-multinode"
    )

    workdir = "Lightning-multinode-templates"
    entrypoint = os.path.join(workdir, script)
    script_name, _ = os.path.splitext(os.path.basename(script))
    job_name = f"mmt-template-test-{prefix}-{script_name.replace('_', '-')}"

    print(f"Starting MMT job {job_name}: {entrypoint}")
    command = f"python {entrypoint}"
    if requirements:
        command = f"pip install -U {requirements} && {command}"
    plugin.run(
        command=command,
        name=job_name,
        cloud_compute=Machine.T4,
        num_instances=2
    )

    studio.stop()

    t0 = t1 = time.time()
    while (t1 - t0) < timeout:
        missing_artifacts = []
        # TODO: also verify size or content
        for artifact in artifacts:
            expected = f"/teamspace/jobs/{job_name}/nodes.0/{artifact}"
            if not os.path.exists(expected):
                missing_artifacts.append(expected)
        if not missing_artifacts:
            break
        t1 = time.time()

    assert not missing_artifacts, f"Artifacts still missing after {timeout}s: {' '.join(missing_artifacts)}"


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main, as_positional=True)
