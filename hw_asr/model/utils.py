available_models = ["QuartzNet5x5", "QuartzNet10x5", "QuartzNet15x5"]


def get_model_config(model_name, n_class):
    if model_name in available_models:
        S_config = int(model_name.split("Net")[1].split("x")[0])
        S = [1] + [int(S_config / 5)] * 5 + [1] * 3
        assert len(S) == 9
    else:
        print(f"Error: {model_name} not implemented use {available_models}")
        raise NotImplementedError
    model_config = {
        "C1": {"R": 1, "K": 33, "C": 256, "S": S[0], "residual": False},
        "B1": {"R": 5, "K": 33, "C": 256, "S": S[1], "residual": True},
        "B2": {"R": 5, "K": 39, "C": 256, "S": S[2], "residual": True},
        "B4": {"R": 5, "K": 63, "C": 512, "S": S[3], "residual": True},
        "B3": {"R": 5, "K": 51, "C": 512, "S": S[4], "residual": True},
        "B5": {"R": 5, "K": 75, "C": 512, "S": S[5], "residual": True},
        "C2": {"R": 1, "K": 87, "C": 512, "S": S[6], "residual": True},
        "C3": {"R": 1, "K": 1, "C": 1024, "S": S[7], "residual": True},
        "C4": {"R": 1, "K": 1, "C": n_class, "S": S[8], "residual": False},
    }
    return model_config


def calculate_output_length(length_in, kernel_size, stride=1, padding=0, dilation=1):
    return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
