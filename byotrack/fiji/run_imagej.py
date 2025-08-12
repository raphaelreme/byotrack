from typing import Union, List
import os
import imagej


class FijiRunnerv2:  # pylint: disable=too-few-public-methods
    """Runs fiji in headless with specified protocol and arguments

    Attributes:
        fiji_path (str | os.PathLike): Path to the fiji executable
            The executable can be found inside the installation folder of Fiji.
            Linux: Fiji.app/ImageJ-<os>
            Windows: Fiji.app/ImageJ-<os>.exe
            MacOs: Fiji.app/Contents/MacOs/ImageJ-<os>
        capture_outputs (bool): Whether to PIPE stderr and stdout into Python
            This will allow you to find the stdout/stderr inside `last_outputs`.
            But outputs are captured, and you do not see them while the scripts run.
            Default: False
        last_outputs (subprocess.CompletedProcess): Outputs of the last subprocess.run

    """

    def __init__(self, fiji_path: Union[str, os.PathLike]) -> None:
        self.fiji_path = fiji_path
        self.runner = imagej.init(fiji_path, mode=imagej.Mode.HEADLESS, add_legacy=True)

        assert os.path.isdir(fiji_path), "Unable to found the given path"

    def run(
        self,
        res_path: Union[str, os.PathLike],
        gt_path: Union[str, os.PathLike],
        n_digit=3,
    ) -> List:
        """Runs fiji with the Biological Measures plugin and the path to both the result and ground truth folder

        Args:
            res_path : Path to the result folder
            gt_path : Path to the ground truth folder
            n_digits: Number of digits in the .tif files

        Returns:
            List: Return the BIO metrics in this order (CT, TF, BCi, CCA)

        """
        args = {"resPath": res_path, "gtPath": gt_path, "noOfDigits": n_digit}
        print("Computing Biological measures with : " + ",".join(f"{key}='{value}'" for key, value in args.items()))

        future = self.runner.command().run("net.celltrackingchallenge.fiji.plugins.plugin_BIOmeasures", True, args)
        module = future.get()

        outputs = module.getOutputs()
        metrics = ["CT", "TF", "BCi", "CCA"]
        if outputs:
            res = [outputs[metric] for metric in metrics]
        else:
            raise RuntimeError(
                "Cannot parse outputs, the CTC software probably found an error"
            )  # Probably not useful, returns (-1,-1,-1,-1) if there is an error

        return res
