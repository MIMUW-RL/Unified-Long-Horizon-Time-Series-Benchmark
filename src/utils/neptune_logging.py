# flake8: noqa

import neptune.new as neptune

USE_NEPTUNE = True

def project():
    raise NotImplementedError("return the project name")

def api_token():
    raise NotImplementedError("return the api token")

def get_neptune_run(params, track_files_path):
    if not USE_NEPTUNE:
        return None


    run = neptune.init_run(
        project=project(),
        api_token=api_token(),
    )

    run["parameters"] = params
    run["checkpoints"].track_files(track_files_path)

    return run


def stop(run):
    if not USE_NEPTUNE:
        return

    run.stop()


def log_w_prefix(run, prefix, results):
    for name, value in results.items():
        log_w_prefix_single(run, prefix, name, value)


def log_w_prefix_single(run, prefix, name, value):
    if run is None:
        return None

    if not USE_NEPTUNE:
        return None

    run["{}/{}".format(prefix, name)].log(value)
