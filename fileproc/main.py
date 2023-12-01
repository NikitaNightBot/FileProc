import numpy as np
import numba
from PIL import Image
import fire
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib as mpl


def readpath(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.byte, sep="")


@numba.njit
def byte_count(data: np.ndarray, remove_null_byte: bool = False) -> np.ndarray:
    counts = np.zeros((256, 256), dtype=np.uint64)
    for index in range(len(data) - 1):
        left = data[index]
        right = data[index + 1]
        counts[left][right] += 1
    if remove_null_byte is True:
        counts[0][0] = 0
    return counts


@numba.njit
def process_normalize_bias(
    data: np.ndarray, bias_power: float, mul: float = 1, remove_null_byte: bool = False
) -> np.ndarray:
    counts = np.float_power(byte_count(data, remove_null_byte), bias_power)
    counts /= counts.max() / mul
    return counts


class Processor:
    @staticmethod
    def bw2(input_path: str, output_path: str) -> None:
        Image.fromarray((byte_count(readpath(input_path)) > 0)).save(  # boolean mask
            output_path
        )

    @staticmethod
    def bw256(
        input_path: str,
        output_path: str,
        bias: float = 4,
        remove_null_byte: bool = True,  # makes the image a bit more lighter due to my normalization fuckups, so turn on usually
    ) -> None:
        Image.fromarray(
            process_normalize_bias(
                readpath(input_path),
                1 / bias,
                mul=255,  # RGB
                remove_null_byte=remove_null_byte,
            ).astype(np.uint8)
        ).save(output_path)

    @staticmethod
    def hue(
        input_path: str,
        output_path: str,
        bias: float = 4,
        remove_null_byte: bool = False,  # not as important here as in bw256 but still nice
    ) -> None:
        hue = process_normalize_bias(
            readpath(input_path),
            1 / bias,
            mul=100,  # HSV
            remove_null_byte=remove_null_byte,
        )
        filler = np.ones((256, 256), dtype=np.uint8) * 100  # :skull:
        Image.fromarray(
            np.dstack((hue, filler, filler)), 
            mode="HSV"
        ).convert("RGB").save(output_path)

    @staticmethod
    def sound_sin(
        input_path: str,
        sample_rate: int,
        output_path: str,
        freqmul: float = 1,
        bias: float = 4,
        remove_null_byte: bool = True,
    ) -> None:
        freq = (
            np.sin(
                process_normalize_bias(
                    readpath(input_path),
                    1 / bias,
                    remove_null_byte=remove_null_byte,
                    mul=freqmul,
                )
            )
            .reshape(256 * 256)
            .astype(np.float32)
        )  # flatten
        wavfile.write(output_path, sample_rate, freq)

    @staticmethod
    def heightmap(
        input_path: str,
        output_path: str,
        colorscheme: str = "Blues",
        bias: float = 4,
        remove_null_byte: bool = True,
    ) -> None:
        freq = process_normalize_bias(
            readpath(input_path),
            1 / bias,
            remove_null_byte=remove_null_byte,
        )
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111, projection="3d")
        x, y = np.meshgrid(range(freq.shape[0]), range(freq.shape[1])) # 256x256 basically always
        ax.plot_surface(x, y, freq, cmap=mpl.colormaps[colorscheme])
        plt.savefig(output_path, dpi=800)

    @staticmethod
    def heatmap(
        input_path: str,
        output_path: str,
        colorscheme: str = "Blues",
        bias: float = 4,
        remove_null_byte: bool = True,
    ) -> None:
        freq = process_normalize_bias(
            readpath(input_path),
            1 / bias,
            remove_null_byte=remove_null_byte,
        )
        plt.figure(frameon=False)
        plt.colorbar(plt.imshow(freq, cmap=mpl.colormaps[colorscheme]))
        plt.savefig(output_path, dpi=800)


if __name__ == "__main__":
    print(
        "\x1B[38;5;93m[INFO]: If you intend to use this multiple times in a session, consider launching this as a repl"
        "[e.g. with the -- --interactive flag, or just import in your repl of choice, and maybe look at the source code :D\x1B[39m]"
    )
    # This is suggested because most of the program's runtime is actually spent on imports and numba compiling the functions when calling them for the first time.

    fire.Fire(Processor)
