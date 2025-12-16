from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment

ws = Workspace.from_config(path="learning_modules/natural-language-processing/azure/workspace_config.json")

env = Environment.from_conda_specification(
    name="nlp-env",
    file_path="environment.yml"
)

src = ScriptRunConfig(
    source_directory=".",
    script="learning_modules/natural-language-processing/pipeline/pipeline.py",
    compute_target="cpu-cluster",
    environment=env
)

exp = Experiment(ws, "nlp_sentiment_pipeline")
run = exp.submit(src)
run.wait_for_completion(show_output=True)
