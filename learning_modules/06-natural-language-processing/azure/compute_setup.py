from azureml.core import Workspace, ComputeTarget, AmlCompute

ws = Workspace.from_config(path="learning_modules/natural-language-processing/azure/workspace_config.json")

compute_name = "cpu-cluster"

if compute_name not in ws.compute_targets:
    config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_DS11_V2",
        max_nodes=2
    )
    cluster = ComputeTarget.create(ws, compute_name, config)
    cluster.wait_for_completion(show_output=True)
else:
    cluster = ws.compute_targets[compute_name]
